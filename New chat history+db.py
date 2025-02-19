import os
import psycopg2  # ✅ PostgreSQL adapter
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage
import time
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ PostgreSQL Connection Details
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "your_database")
DB_USER = os.getenv("POSTGRES_USER", "your_username")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "your_password")

# Initialize Embeddings and FAISS Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Define memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# Initialize the tokenizer (Hugging Face tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

def get_db_connection():
    """✅ Establish a PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def query_database(user_message):
    """✅ Retrieve relevant information from PostgreSQL based on user input."""
    conn = get_db_connection()
    if not conn:
        return "Error: Unable to connect to database."

    try:
        cursor = conn.cursor()

        # Example Query: Searching for relevant topics
        cursor.execute("SELECT info FROM knowledge_base WHERE topic ILIKE %s LIMIT 5;", ('%' + user_message + '%',))
        results = cursor.fetchall()
        conn.close()

        if results:
            return "\n".join(row[0] for row in results)
        return "No relevant database information found."

    except Exception as e:
        print(f"❌ Database query error: {e}")
        return "Error retrieving data."

def tokenize_and_chunk(text, max_tokens=512):
    """✅ Tokenize and chunk long texts before adding to memory."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def retrieve_past_context(user_message, k=15):
    """✅ Retrieve relevant past conversation context."""
    docs = vector_store.similarity_search(user_message, k=k)

    if not docs:
        print(f"⚠️ No relevant past context found for: {user_message}")
        return []

    # Sort retrieved messages by recency (if timestamp exists)
    sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get("timestamp", 0), reverse=True)
    return [doc.page_content for doc in sorted_docs]

def manage_memory():
    """✅ Manages conversation history by pruning old memories."""
    all_memory = memory.load_memory_variables({}).get("chat_history", [])

    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))

    return "\n".join(memory_chunks[-5:])  # Keep the last 5 chunks

# Chatbot loop
print("🤖 Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve past context
    past_context = retrieve_past_context(question)
    context_str = "\n".join(past_context)

    # Retrieve database info
    db_info = query_database(question)

    # Load memory context
    memory_context = manage_memory()

    # Format final prompt
    prompt = prompt_.format(question=question, context=context_str + "\n" + memory_context + "\nDatabase Info:\n" + db_info)

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

    # Save conversation to memory
    memory.save_context({"User": question}, {"AI": response_text})

    # ✅ Store user input and response in FAISS for future retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]  # Store with a timestamp
    )
