# LangChainAgents

A small collection of LangChain/LangGraph examples:

- ReAct agent with tools and memory (`main.py`)
- Minimal LangGraph agent (`langgraph.py`)
- Simple RAG pipeline with vector store (`rag_example.py`)
- Multimodal (vision + text) prompt to an LLM (`multimodal_example.py`)

These scripts are meant to be run independently and demonstrate focused patterns.

## Prerequisites

- Python 3.10+ recommended
- API keys (set via environment variables or a local `.env` file):
  - `GROQ_API_KEY` (for Groq-hosted LLMs in `main.py`, `rag_example.py`, `multimodal_example.py`)
  - `TAVILY_API_KEY` (for web search tool in `main.py`)
  - `LANGSMITH_API_KEY` (optional; for tracing/observability if you use LangSmith)
  - `ANTHROPIC_API_KEY` (only if you change/use Anthropic models as in `langgraph.py`)
  - `USER_AGENT` (used by `rag_example.py` to fetch web pages politely)

## Install

From this folder, install Python dependencies:

```powershell
# From Windows PowerShell
cd "c:\Users\jonah\Code\Machine Learning\Deep learning\Agents\LangChainAgents"
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```