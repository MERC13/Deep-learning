from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import base64
import httpx

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = init_chat_model("meta-llama/llama-4-maverick-17b-128e-instruct", model_provider="groq")

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

# Compose a HumanMessage with content parts using allowed types: 'text' and 'image_url'
message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        # You can also pass the public URL directly via image_url -> { url: <http-url> }
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ]
)

response = model.invoke([message])
print(response.content)