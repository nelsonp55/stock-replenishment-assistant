# Stock Replenishment Chatbot

Final project for the bootcamp (End to End Data Science Project).

## Project Goal

Stock management system with a chatbot interface, simulating the replenishment flow between headquarters (central warehouse) and stores.

**Core flow:**
1. The user requests, via chatbot, the replenishment of a specific brand for a specific store.
2. The system checks available stock at headquarters.
3. If stock is available, it generates a picking/separation PDF document.

The system is **stateless by design** — there are no database writes. The generated PDF is the output artifact itself.

## Dataset

Real product data (brand, reference, size, location, quantity), with anonymized store names. The dataset distinguishes three warehouse types:
- **Warehouse 1** — Headquarters (stock source for replenishment)
- **Warehouse 2** — Stores (core of the business)
- **Warehouse 5** — Outlets (selling old stock)

## Repository Structure

```
FinalProject/
├── notebooks/     # exploration and EDA
├── data/          # dataset (not versioned - see .gitignore)
├── src/           # reusable Python code (replenishment logic, PDF generation, chatbot)
├── outputs/       # generated picking PDFs (not versioned)
├── requirements.txt
└── README.md
```

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 1 | Aug 18–24 | Data cleaning and EDA |
| 2 | Aug 25–31 | Replenishment logic in Python (no chatbot) |
| 3 | Sep 1–5 | PDF generation + basic Streamlit dashboard |
| 4 | Sep 6–9 | Chatbot v1 with structured commands (function calling) |
| 5 | Sep 10–12 | Documentation, GitHub organization, slides, presentation rehearsal |

## How to Run the Project

_(to be filled in as the project progresses)_

## Author

Nelson — Ironhack Bootcamp, Data Science
