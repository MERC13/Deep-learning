# Simple LangChaing agent with memory and tools

# Imports and Setup
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

search = TavilySearch(max_results=2)
tools = [search]
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
model_with_tools = model.bind_tools(tools)
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# Usage examples

for step in agent_executor.stream(
    {"messages": [("user", "Hi, I'm Bob!"), ("user", "What is my name?"), ("user", "What is the weather like today?")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
