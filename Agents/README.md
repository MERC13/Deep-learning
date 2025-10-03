# Agents

A grab‑bag of small agentic AI examples and a fuller research assistant pipeline. Each subfolder is self‑contained—pick what matches the pattern you want to explore.

## Contents

| Subfolder | Focus | Stack Highlights |
|-----------|-------|------------------|
| `AutogenAgents/` | Lightweight multi‑agent chat with tool (agent-as-tool) chaining | Microsoft AutoGen (agentchat), local Ollama models |
| `LangChainAgents/` | Assorted LangChain / LangGraph patterns (ReAct, LangGraph state machine, RAG, multimodal) | LangChain, LangGraph, Groq / Ollama, Tavily search |
| `research-assistant/` | End‑to‑end PDF ingestion → embedding → retrieval → LLM summarization → email digest | PyMuPDF, Sentence-Transformers, Pinecone, Ollama, SendGrid |

## Quick Start (Pick One)

### 1. Autogen basic multi‑agent demo
```
cd Agents/AutogenAgents
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Ensure Ollama is running locally with `ollama pull llama3.2`
python basic.py
```

### 2. LangChain / LangGraph examples
```
cd Agents/LangChainAgents
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Set needed API keys (GROQ_API_KEY, TAVILY_API_KEY, etc.)
python main.py               # ReAct agent
python langgraph.py          # Minimal LangGraph
python rag_example.py        # Simple RAG
python multimodal_example.py # Vision + text prompt
```

### 3. Research Assistant pipeline
See the detailed `research-assistant/README.md` for full instructions (env vars, Docker, scheduling). Minimal local run:
```
cd Agents/research-assistant
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env_sample .env  # then edit keys
# Start Ollama & pull a model if using LLM summaries
$env:PYTHONPATH = 'src'
python -m tools.print_digest
```