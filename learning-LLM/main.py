from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlite3
import threading

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")

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
    """Build the prompt with active preferences."""
    prefs = get_all_suggestions()
    if prefs:
        return f"{prefs}\nUser: {user_input}\nChatbot:"
    else:
        return f"User: {user_input}\nChatbot:"

def generate_response(prompt: str) -> str:
    """Generate chatbot reply using the pretrained model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Chatbot:")[-1].strip()

# ----------------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "").strip()
    
    if text.lower().startswith("suggestion:"):
        # Format: "Suggestion: category=visualization; use charts"
        print("---Parsing suggestion---")
        _, body = text.split(":", 1)
        category, instruction = body.split(";", 1)
        add_suggestion(category.strip(), instruction.strip())
        return jsonify({"reply": "👍 Preference noted and saved."})
    
    prompt = construct_prompt(text)
    reply = generate_response(prompt)
    print("RESPONSE:", reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)