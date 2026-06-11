# AI-ABSA: Evaluative Analysis of Generative AI Models

AI-ABSA (Aspect-Based Sentiment Analysis) is a specialized sentiment analysis system designed for the evaluative analysis of Large Language Models (LLMs) such as ChatGPT, Claude, Gemini, and DeepSeek. The project follows an incremental development strategy, transitioning from a basic pipeline to a state-of-the-art (SOTA) ASTE architecture.

---

## Project Objectives
- **Tier 1:** Implementation of a baseline pipeline (Aspect Extraction and Sentiment Classification).
- **Tier 2:** Upgrade to a Generative ABSA (ASTE) architecture for Aspect-Opinion-Sentiment triplet extraction.
- **Scaling:** Mass inference on 1 million real-world conversations from the LMSYS Chatbot Arena dataset.

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **Deep Learning:** PyTorch (MPS Acceleration for Apple Silicon), Transformers
- **Validation:** Pydantic (Data Contract)
- **Interface:** FastAPI & Streamlit

## Installation and Setup

### 1. Prerequisites
Install the `uv` package manager:
```bash
curl -LsSf https://astral-sh/uv/install.sh | sh
```

### 2. Project Initialization
Synchronize the environment and install dependencies:
```bash
uv sync
```

### 3. Execution
Launch the analysis dashboard:
```bash
uv run streamlit run src/app.py
```

## Project Structure
```text
ai-absa/
├── data/               # Raw and processed datasets (Mendeley, LMSYS)
├── docs/               # Technical documentation & Data Contract
├── src/
│   ├── data/           # Data loaders and Pydantic schemas
│   ├── models/         # Basic (Tier 1) and Advanced (Tier 2) models
│   ├── training/       # Training scripts & Fine-tuning logic
│   └── utils/          # Device configuration (MPS) and logging
├── pyproject.toml      # Project configuration (uv)
└── README.md
```

## Data Governance
The project strictly adheres to Pydantic schemas to ensure data consistency across all modules and data report. Detailed specifications can be found at: [docs/data_contract.md](docs/data_contract.md) & [docs/data_report.md](docs/data_report.md).

