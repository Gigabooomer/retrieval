import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ PostgreSQL Connection using SQLAlchemy
DB_URL = os.getenv("DATABASE_URL", "postgresql://your_user:your_password@localhost:5432/your_database")
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

# Initialize Embeddings and FAISS Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Paths for template and schema files
TEMPLATE_FILE = "temp_path"
SCHEMA_FILE = "context.sql"

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

# ✅ Directly provided chat history
chat_context = [
    {"user": "What is the capital of Japan?", "assistant": "The capital of Japan is Tokyo."},
    {"user": "Who is the prime minister?", "assistant": "The prime minister of Japan is Fumio Kishida."},
]

def retrieve_past_context(user_message, k=15):
    """✅ Retrieve relevant past conversation context using FAISS."""
    docs = vector_store.similarity_search(user_message, k=k)
    return [doc.page_content for doc in docs] if docs else []

def format_chat_history(chat_context, max_turns=5):
    """✅ Convert recent chat history into a properly formatted string."""
    recent_context = chat_context[:max_turns]  # Take the last N messages
    formatted_history = "\n".join(f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in recent_context)
    return formatted_history

# Chatbot loop
print("🤖 Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve past conversation context
    past_context = retrieve_past_context(question)
    context_str = "\n".join(past_context)

    # ✅ Extract most recent chat history from `chat_context`
    memory_context = format_chat_history(chat_context)

    # Format final prompt
    prompt = prompt_.format(
        question=question,
        context=context_str + "\n" + memory_context
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

    # ✅ Update chat_context with the new user-bot exchange
    chat_context.insert(0, {"user": question, "assistant": response_text})

    # ✅ Store in FAISS for future similarity-based retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )
