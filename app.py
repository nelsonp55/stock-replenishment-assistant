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

# last stock listing, kept so it can be exported to Excel
if 'last_stock_detail' not in st.session_state:
    st.session_state.last_stock_detail = None


# --------------------------------------------------------------------------
# Tool definitions exposed to the model
# --------------------------------------------------------------------------

STORE_DESCRIPTION = (
    "Store name, one of: 'Loja Amoreiras', 'Loja Colombo', 'Outlet Cascais', "
    "'Outlet Gaia', 'Outlet Loures', 'Outlet Almada'. Use 'HQ' for the central "
    "warehouse. Pass whatever place name the user typed, even a short form such as "
    "'Amoreiras' or 'Gaia' — never put a place name in a brand argument."
)

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'find_replenishable_sizes',
            'description': (
                'Find sizes the central warehouse has in stock but a store does not, '
                'for references the store already carries. Use this whenever the user '
                'asks what can be replenished or what sizes are missing. Leave '
                'store_input out to cover every store at once — do not call this tool '
                'once per store.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION + ' Omit for all stores.'},
                    'brand_input': {'type': 'string', 'description': 'Brand name. Omit to cover all brands.'},
                    'reference_input': {'type': 'string', 'description': 'Optional reference code or prefix.'},
                    'units_per_size': {'type': 'integer', 'description': 'Units to send per missing size, if the user states one.'},
                },
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
            'name': 'get_stock_detail',
            'description': (
                'List the actual stock held, one row per reference and size, with '
                'quantities. Use when the user asks to see the stock in detail, the '
                'references held, or a full listing rather than a total. Leave '
                'store_input out to cover every store at once — do not call this tool '
                'once per store.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'store_input': {'type': 'string', 'description': STORE_DESCRIPTION + ' Omit for all stores.'},
                    'brand_input': {'type': 'string', 'description': 'Optional brand name.'},
                    'reference_input': {'type': 'string', 'description': 'Optional reference code or prefix.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_quantity',
            'description': (
                'Stock totals, and which brands a store carries. Use for questions like '
                '"how much stock does store X have", "how many units of brand Y", or '
                '"which brands exist in store X". Without a brand it breaks the total '
                'down by brand for that store.'
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
    "Never mention a table, a list below, or anything shown elsewhere on the screen: "
    "you cannot see the interface. Just give the answer.\n\n"
    "Apply only the filters stated in the user's latest message. If they do not name "
    "a brand, cover all brands; if they do not name a store, cover all stores. Never "
    "carry a brand or store over from an earlier question unless the user refers back "
    "to it.\n\n"
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
            store_input=args.get('store_input'),
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

    if name == 'get_stock_detail':
        result = rp.get_stock_detail(
            df,
            store_input=args.get('store_input'),
            brand_input=args.get('brand_input'),
            reference_input=args.get('reference_input'),
        )
        if result.empty:
            return 'No stock found with those filters.', None

        summary = (
            f"{len(result)} rows, {int(result['QUANTITY'].sum())} units, "
            f"across {result['REFERENCE'].nunique()} references."
        )
        return summary, result

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

    # this turn replaces whatever the generation panel was showing
    st.session_state.last_replenishment = None
    st.session_state.last_stock_detail = None

    for call in message.tool_calls:
        import json
        args = json.loads(call.function.arguments)
        try:
            text, result_df = run_tool(call.function.name, args)
        except ValueError as e:
            text, result_df = f'Error: {e}', None

        if result_df is not None:
            table = result_df

        if result_df is not None and call.function.name == 'get_stock_detail':
            st.session_state.last_stock_detail = {
                'store': args.get('store_input'),
                'brand': args.get('brand_input'),
                'rows': len(result_df),
            }

        # only a replenishment query may enable the generation panel
        if result_df is not None and call.function.name == 'find_replenishable_sizes':
            st.session_state.last_replenishment = {
                'store': args.get('store_input'),
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

BANNER = Path(__file__).resolve().parent / 'assets' / 'banner.png'


@st.cache_data
def banner_uri():
    """Read the header image once and inline it, so CSS can position text over it."""
    import base64
    return 'data:image/png;base64,' + base64.b64encode(BANNER.read_bytes()).decode()


CSS = """
<style>
    /* header banner with the title sitting on top of it */
    .app-header {
        background-image: url('__BANNER__');
        background-size: cover;
        background-position: center right;
        border-radius: 12px;
        padding: 2.3rem 2rem;
        margin-bottom: 1.6rem;
        border: 1px solid #333C4A;
    }
    .app-header h1 {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.01em;
    }
    /* chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #232B38;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    /* the chat box is the main control, so give it presence */
    [data-testid="stChatInput"] {
        border: 1px solid #46536633;
        border-radius: 12px;
        background-color: #232B38;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #82C7A5;
        box-shadow: 0 0 0 2px rgba(130, 199, 165, 0.18);
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1.1rem !important;
        min-height: 3.6rem !important;
        padding-top: 1.05rem !important;
        padding-bottom: 1.05rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #7C8899 !important;
    }
    /* sidebar section labels */
    .side-label {
        color: #8B96A6;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .chip {
        display: block;
        background-color: #1B212C;
        border: 1px solid #333C4A;
        border-radius: 6px;
        padding: 0.42rem 0.6rem;
        margin-bottom: 0.36rem;
        color: #C3CCD9;
        font-size: 0.79rem;
        line-height: 1.35;
    }
    .chip-muted {
        color: #8B96A6;
        font-style: italic;
    }
    .panel-title {
        color: #82C7A5;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    footer, #MainMenu {visibility: hidden;}
</style>
"""

st.markdown(CSS.replace('__BANNER__', banner_uri()), unsafe_allow_html=True)

st.markdown(
    '<div class="app-header">'
    '<h1>Stock Replenishment Assistant</h1>'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="side-label">Stores</div>', unsafe_allow_html=True)
    store_labels = [rp.STORE_NAMES[s] for s in rp.SELECTED_STORES if s in rp.STORE_NAMES]
    for label in store_labels:
        st.markdown(f'<div class="chip">{label}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chip chip-muted">HQ — central warehouse (source)</div>',
                unsafe_allow_html=True)

    brands = sorted(df['BRAND'].dropna().unique())
    with st.expander(f'Brands ({len(brands)})'):
        st.markdown(
            ''.join(f'<div class="chip">{b}</div>' for b in brands),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
    st.metric('Sizes committed', len(st.session_state.allocated))
    st.caption('Stock is only committed when a document is generated. Queries never allocate anything.')

    if st.button('Reset allocations', use_container_width=True):
        st.session_state.allocated = {}
        st.session_state.last_replenishment = None
        st.rerun()

    st.markdown('<div class="side-label">Try asking</div>', unsafe_allow_html=True)
    for example in [
        'What can I replenish in Loja Amoreiras?',
        'Which Nike sizes are missing in Loja Colombo?',
        'Where is reference DH2920?',
        'How much stock does Outlet Gaia have?',
    ]:
        st.markdown(f'<div class="chip">{example}</div>', unsafe_allow_html=True)

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

# exporting the current stock listing
stock = st.session_state.last_stock_detail
if stock:
    st.divider()
    st.markdown('<div class="panel-title">Export this stock listing</div>', unsafe_allow_html=True)
    st.caption(
        (stock['store'] or 'All stores')
        + (f" · {stock['brand']}" if stock['brand'] else '')
        + f" · {stock['rows']} rows"
    )
    if st.button('Export to Excel', type='primary'):
        try:
            path = rp.export_stock(
                df,
                store_input=stock['store'],
                brand_input=stock['brand'],
            )
            st.success(f'Saved: {path}')
            with open(path, 'rb') as f:
                st.download_button(
                    'Download',
                    data=f.read(),
                    file_name=Path(path).name,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
        except ValueError as e:
            st.error(str(e))

# document generation is a separate, explicit confirmation step
last = st.session_state.last_replenishment
if last:
    st.divider()
    st.markdown('<div class="panel-title">Generate picking document</div>', unsafe_allow_html=True)
    st.caption(
        (last['store'] or 'All stores')
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
