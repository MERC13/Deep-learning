# LangChainAgents

A small collection of LangChain / LangGraph examples:

- ReAct agent with tools and memory (`main.py`)
- Minimal LangGraph agent (`langgraph.py`)
- Simple RAG pipeline with vector store (`rag_example.py`)
- Multimodal (vision + text) prompt to an LLM (`multimodal_example.py`)

These scripts are meant to be run independently and demonstrate focused patterns.

## Prerequisites

- Python 3.10+ recommended
- API keys (set via environment variables or a local `.env` file):
  - `GROQ_API_KEY` (Groq-hosted LLMs used by some examples)
  - `TAVILY_API_KEY` (optional; for web search tool)
  - `LANGSMITH_API_KEY` (optional; observability/tracing)
  - `ANTHROPIC_API_KEY` (only if you swap in Anthropic models)
  - `USER_AGENT` (used by `rag_example.py` when fetching web pages)

## Install

From this folder, install Python dependencies (PowerShell):

```powershell
cd Agents/LangChainAgents
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```