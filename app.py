"""Streamlit chat interface for the stock replenishment assistant.

The language model is used only to route the user's question to one of the
deterministic query functions in src/replenishment.py. It never generates stock
figures itself: every number shown comes from the dataset.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import replenishment as rp

MODEL = 'gpt-4o-mini'

st.set_page_config(page_title='Stock Replenishment Assistant', page_icon='👟', layout='wide')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner='Loading stock data...')
def get_data():
    """Load and clean the dataset once, then reuse it across reruns."""
    return rp.load_data()


df = get_data()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

# chat history shown in the interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# stock committed to documents generated in this session, {(REFERENCE, SIZE): qty}
if 'allocated' not in st.session_state:
    st.session_state.allocated = {}

# last replenishment result, kept so the document can be generated from it
if 'last_replenishment' not in st.session_state:
    st.session_state.last_replenishment = None


# --------------------------------------------------------------------------
# Tool definitions exposed to the model
# --------------------------------------------------------------------------

STORE_DESCRIPTION = (
    "Store name, one of: 'Loja Amoreiras', 'Loja Colombo', 'Outlet Cascais', "
    "'Outlet Gaia', 'Outlet Loures', 'Outlet Almada'. Use 'HQ' for the central warehouse."
)

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'find_replenishable_sizes',
            'description': (
                'Find sizes the central warehouse has in stock but a store does not, '
                'for references the store already carries. Use this whenever the user '
                'asks what can be replenished or what sizes are missing in a store.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION},
                    'brand_input': {'type': 'string', 'description': 'Optional brand name to narrow the search.'},
                    'reference_input': {'type': 'string', 'description': 'Optional reference code or prefix.'},
                    'units_per_size': {'type': 'integer', 'description': 'Units to send per missing size, if the user states one.'},
                },
                'required': ['store_input'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'find_reference',
            'description': (
                'Look up where a specific reference code exists across the warehouse and '
                'stores, with quantities by size. Matches by prefix.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'reference_input': {'type': 'string', 'description': 'Reference code or prefix, e.g. DH2920.'},
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION},
                    'size_input': {'type': 'string', 'description': 'Optional size to filter by.'},
                },
                'required': ['reference_input'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_quantity',
            'description': (
                'Total stock quantities, without a size breakdown. Use for questions like '
                '"how much stock does store X have" or "how many units of brand Y".'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION},
                    'brand_input': {'type': 'string', 'description': 'Optional brand name.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'find_model_by_type',
            'description': (
                'Show where a brand or reference exists, grouped by location type '
                '(central warehouse, stores, outlets).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'reference_input': {'type': 'string', 'description': 'Optional reference code or prefix.'},
                    'brand_input': {'type': 'string', 'description': 'Optional brand name.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'export_stock',
            'description': (
                'Export the full stock of one or more stores to an Excel file, as a size grid. '
                'Use only when the user explicitly asks for an export or a stock file.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION},
                    'brand_input': {'type': 'string', 'description': 'Optional brand name.'},
                },
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a stock replenishment assistant for a footwear retailer. You help store "
    "managers check what can be replenished from the central warehouse (HQ).\n\n"
    "Always use the provided tools to answer questions about stock. Never invent, "
    "estimate, aggregate or round any figure: every number must appear exactly as the "
    "tool returned it.\n\n"
    "When a tool returns text, show that text to the user as it is. Do not rewrite it "
    "into ranges, approximations or summaries, and do not describe quantities with "
    "words like 'several' or 'various'. Add at most one short sentence of your own "
    "before or after it.\n\n"
    "When a tool returns a row count instead of the rows themselves, report those "
    "figures and tell the user the table is shown below the message. Never mention a "
    "table in any other situation.\n\n"
    "If a store or brand name is not recognised, say so and list the valid options "
    "from the error message."
)


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

def run_tool(name, args):
    """Run a query function and return (text_for_model, dataframe_or_None).

    Only the text is sent back to the model; dataframes are displayed directly so
    large results never consume tokens.
    """
    if name == 'find_replenishable_sizes':
        result = rp.find_replenishable_sizes(
            df,
            store_input=args['store_input'],
            brand_input=args.get('brand_input'),
            reference_input=args.get('reference_input'),
            allocated=st.session_state.allocated,
            units_per_size=args.get('units_per_size'),
        )
        if result.empty:
            return 'Nothing to replenish with those filters.', None

        summary = (
            f"{len(result)} lines to replenish, {int(result['SEND_QTY'].sum())} units in total, "
            f"across {result['REFERENCE'].nunique()} references."
        )
        return summary, result

    if name == 'find_reference':
        return rp.find_reference(
            df,
            reference_input=args['reference_input'],
            store_input=args.get('store_input'),
            size_input=args.get('size_input'),
        ), None

    if name == 'get_quantity':
        return rp.get_quantity(
            df,
            store_input=args.get('store_input'),
            brand_input=args.get('brand_input'),
        ), None

    if name == 'find_model_by_type':
        return rp.find_model_by_type(
            df,
            reference_input=args.get('reference_input'),
            brand_input=args.get('brand_input'),
        ), None

    if name == 'export_stock':
        path = rp.export_stock(
            df,
            store_input=args.get('store_input'),
            brand_input=args.get('brand_input'),
        )
        return f'Stock exported to {path}', None

    return f'Unknown tool: {name}', None


def ask_model(client, user_message):
    """Send the conversation to the model, run any tool it calls, and return the reply."""
    history = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        history.append({'role': m['role'], 'content': m['content']})
    history.append({'role': 'user', 'content': user_message})

    response = client.chat.completions.create(
        model=MODEL, messages=history, tools=TOOLS, tool_choice='auto'
    )
    message = response.choices[0].message

    if not message.tool_calls:
        return message.content, None

    history.append(message)
    table = None

    for call in message.tool_calls:
        import json
        args = json.loads(call.function.arguments)
        try:
            text, result_df = run_tool(call.function.name, args)
        except ValueError as e:
            text, result_df = f'Error: {e}', None

        if result_df is not None:
            table = result_df
            st.session_state.last_replenishment = {
                'store': args['store_input'],
                'brand': args.get('brand_input'),
                'reference': args.get('reference_input'),
                'units_per_size': args.get('units_per_size'),
                'rows': len(result_df),
            }

        history.append({
            'role': 'tool',
            'tool_call_id': call.id,
            'content': text,
        })

    follow_up = client.chat.completions.create(model=MODEL, messages=history)
    return follow_up.choices[0].message.content, table


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

st.title('👟 Stock Replenishment Assistant')
st.caption('Ask what can be replenished from the central warehouse to a store.')

with st.sidebar:
    st.subheader('Session')
    st.metric('Reference/size pairs committed', len(st.session_state.allocated))
    st.caption(
        'Stock is only committed when a document is generated. Queries never '
        'allocate anything.'
    )
    if st.button('Reset session allocations'):
        st.session_state.allocated = {}
        st.session_state.last_replenishment = None
        st.rerun()

    st.divider()
    st.subheader('Examples')
    st.code(
        'What can I replenish in Loja Amoreiras?\n'
        'Which Nike sizes are missing in Loja Colombo?\n'
        'Where is reference DH2920?\n'
        'How much stock does Outlet Gaia have?',
        language=None,
    )

if not st.secrets.get('OPENAI_API_KEY'):
    st.error('No OPENAI_API_KEY found. Add it to .streamlit/secrets.toml.')
    st.stop()

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# replay the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('table') is not None:
            st.dataframe(message['table'], use_container_width=True, hide_index=True)

if prompt := st.chat_input('Ask about stock or replenishment...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Checking stock...'):
            try:
                reply, table = ask_model(client, prompt)
            except Exception as e:
                reply, table = f'Something went wrong: {e}', None

        st.markdown(reply)
        if table is not None:
            st.dataframe(table, use_container_width=True, hide_index=True)

    st.session_state.messages.append({'role': 'assistant', 'content': reply, 'table': table})

# document generation is a separate, explicit confirmation step
last = st.session_state.last_replenishment
if last:
    st.divider()
    st.subheader('Generate picking document')
    st.caption(
        f"Last query: {last['store']}"
        + (f" · {last['brand']}" if last['brand'] else '')
        + (f" · {last['reference']}" if last['reference'] else '')
        + f" · {last['rows']} lines"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        units = st.number_input('Units per size', min_value=1, max_value=20, value=1)
    with col2:
        st.write('')
        st.write('')
        if st.button('Generate document', type='primary'):
            try:
                filename, new_allocated = rp.generate_replenishment_document(
                    df,
                    store_input=last['store'],
                    brand_input=last['brand'],
                    reference_input=last['reference'],
                    allocated=st.session_state.allocated,
                    units_per_size=units,
                )
                # stock leaves HQ only here
                st.session_state.allocated = new_allocated
                st.success(f'Document saved: {filename}')

                with open(filename, 'rb') as f:
                    st.download_button(
                        'Download',
                        data=f.read(),
                        file_name=Path(filename).name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
            except ValueError as e:
                st.error(str(e))
