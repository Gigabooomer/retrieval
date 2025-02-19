import os
import time
import json  # ✅ Added to read the provided file
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import AIMessage, HumanMessage
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Embeddings and FAISS Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Define memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Paths for template, schema, and chat history files
TEMPLATE_FILE = "temp_path"
SCHEMA_FILE = "context.sql"
CHAT_HISTORY_FILE = "chat_history.json"  # ✅ File containing previous conversations

# Remote LLM API
from config import LLM_2_API
local_llm = RemoteRunnable(LLM_2_API)

# Load template
with open(TEMPLATE_FILE, "r") as f:
    template = f.read().strip()

# Load SQL schema
with open(SCHEMA_FILE, "r") as f:
    schema = f.read().strip()

# Define Prompt Template
prompt_ = PromptTemplate.from_template(template)

# Initialize the tokenizer (Hugging Face tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

# ✅ Load chat history from file
def load_chat_history(file_path):
    """Load past chat history from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)  # Expecting a list of {"user": "...", "assistant": "..."}
    return []

chat_context = load_chat_history(CHAT_HISTORY_FILE)

# ✅ Function to manually tokenize and chunk text
def tokenize_and_chunk(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# ✅ Function to retrieve relevant past context
def retrieve_past_context(user_message, k=15):
    docs = vector_store.similarity_search(user_message, k=k)
    return [doc.page_content for doc in docs] if docs else []

# ✅ Function to format the last few stored conversations
def format_chat_history(chat_context, max_turns=5):
    recent_context = chat_context[:max_turns]  # Take the last N messages
    return "\n".join(f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in recent_context)

# ✅ Manual memory management
def manage_memory():
    """Fetch conversation history, tokenize, chunk, and retain recent messages."""
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    return "\n".join(memory_chunks[-5:])  # Keep the last 5 chunks

# Chatbot loop
print("Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve past context using FAISS
    past_context = retrieve_past_context(question)
    context_str = "\n".join(past_context)

    # Load memory from file + manage in-memory conversation history
    memory_context = manage_memory()
    file_chat_context = format_chat_history(chat_context)

    # Format final prompt
    prompt = prompt_.format(
        question=question,
        context=context_str + "\n" + file_chat_context + "\n" + memory_context
    )

    # Debugging: Print the final prompt before sending to LLM
    print("\n--- Final Prompt ---")
    print(prompt)
    print("--------------------\n")

    # Generate response from LLM
    response_text = ""
    for chunk in local_llm.stream(prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")  # Newline for better readability

    # ✅ Update chat_context with the new exchange and save it to the file
    new_entry = {"user": question, "assistant": response_text}
    chat_context.insert(0, new_entry)  # Add newest message to the front
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_context, f, ensure_ascii=False, indent=4)

    # ✅ Store in FAISS for future retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )
