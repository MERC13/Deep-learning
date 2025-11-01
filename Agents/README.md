# Agents

A set of agent-focused examples and a research-assistant pipeline. Each subfolder is intended to be run independently; pick the example that matches what you want to explore.

## Contents

| Subfolder | Focus | Stack highlights |
|-----------|-------|------------------|
| `AutogenAgents/` | Lightweight multi-agent chat with tool chaining | Microsoft AutoGen, local Ollama models |
| `LangChainAgents/` | LangChain / LangGraph patterns (ReAct, RAG, multimodal) | LangChain, LangGraph, Groq/Ollama |
| `research-assistant/` | End-to-end PDF ingestion → embedding → summarization → email digest | PyMuPDF, sentence-transformers, Pinecone, Ollama, SendGrid |

## Quick start (pick one)

### 1) Autogen basic multi-agent demo

From PowerShell:

```powershell
cd Agents/AutogenAgents
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Ensure Ollama is running locally (if used): ollama pull <model>
python basic.py
```

### 2) LangChain / LangGraph examples

From PowerShell:

```powershell
cd Agents/LangChainAgents
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set API keys in env vars or .env (see LangChainAgents/README.md)
python main.py               # ReAct agent
python langgraph.py          # Minimal LangGraph
python rag_example.py        # Simple RAG
python multimodal_example.py # Vision + text example
```

### 3) Research Assistant pipeline

See `Agents/research-assistant/README.md` for full instructions. Minimal local run:

```powershell
cd Agents/research-assistant
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env_sample .env  # then edit keys in .env
# Start Ollama & pull a model if using LLM summaries (optional)
$env:PYTHONPATH = 'src'
python -m tools.print_digest
```