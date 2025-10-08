from flask import Flask, request, jsonify
from openai import OpenAI
import sqlite3
import threading

app = Flask(__name__)

# Initialize OpenAI client for Ollama local server
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama local API endpoint
    api_key="ollama"                        # Dummy key for auth as required by Ollama
)

conn = sqlite3.connect("suggestions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS suggestions (
    category TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
""")
conn.commit()
db_lock = threading.Lock()

# ----------------------------------------------------------------------------------

def add_suggestion(category: str, text: str):
    """Store or update a user suggestion."""
    with db_lock:
        cursor.execute("""
            INSERT INTO suggestions(category, text)
            VALUES (?, ?)
            ON CONFLICT(category) DO UPDATE SET text=excluded.text
        """, (category, text))
        conn.commit()

def get_all_suggestions() -> str:
    """Retrieve and summarize all suggestions."""
    with db_lock:
        cursor.execute("SELECT text FROM suggestions")
        entries = cursor.fetchall()
    return "\n".join(entry[0] for entry in entries)

def construct_prompt(user_input: str) -> str:
    """Build prompt with active preferences included as system message."""
    prefs = get_all_suggestions()
    if prefs:
        system_content = prefs
    else:
        system_content = "You are a helpful assistant."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]

def generate_response_ollama(messages):
    """Generate chatbot reply using Ollama GPT-OSS API."""
    response = client.chat.completions.create(
        model="gpt-oss:20b",  # Use your pulled Ollama GPT-OSS model version here
        messages=messages
    )
    return response.choices[0].message.content

# ----------------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "").strip()
    
    if text.lower().startswith("suggestion:"):
        # Format: "Suggestion: category=visualization; use charts"
        print("---Parsing suggestion---")
        try:
            _, body = text.split(":", 1)
            category, instruction = body.split(";", 1)
            add_suggestion(category.strip(), instruction.strip())
            return jsonify({"reply": "👍 Preference noted and saved."})
        except Exception as e:
            return jsonify({"reply": f"Error parsing suggestion: {str(e)}"}), 400
    
    messages = construct_prompt(text)
    reply = generate_response_ollama(messages)
    print("RESPONSE:", reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
