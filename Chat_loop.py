import os
import time
import faiss
import numpy as np
import psycopg2
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS with HNSW Index
d = 384  # Dimension size for MiniLM embeddings
index = faiss.IndexHNSWFlat(d, 32)  
vector_store = FAISS(embedding_function=embeddings, index=index)

# Define memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Remote LLM API
from config import LLM_2_API
local_llm = RemoteRunnable(LLM_2_API)

# PostgreSQL Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")

def get_chat_history(user_id):
    """Retrieve chat history from the PostgreSQL database using a user ID."""
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()
    cursor.execute("SELECT user_message, assistant_response FROM chat_history WHERE user_id = %s ORDER BY timestamp ASC;", (user_id,))
    history = cursor.fetchall()
    cursor.close()
    connection.close()

    return "\n".join([f"User: {row[0]}\nAssistant: {row[1]}" for row in history]) if history else ""

# Load template and schema
TEMPLATE_FILE = "temp_path"
SCHEMA_FILE = "context.sql"

with open(TEMPLATE_FILE, "r") as f:
    template = f.read().strip()

with open(SCHEMA_FILE, "r") as f:
    database_schema = f.read().strip()

# Define Prompt Template
prompt_ = PromptTemplate.from_template(f"""
You are an AI assistant helping users with SQL queries and database-related questions.

### Database Schema:
{database_schema}

### Context:
{{context}}

### User Question:
{{question}}

### Answer:
""")

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

def generate_standalone_question(chat_history, user_input):
    """Rewrites user input into a standalone question."""
    standalone_question_prompt = f"""
    You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.

    ### Chat History:
    {chat_history}

    ### User Input:
    {user_input}

    ### Standalone Question:
    """
    rephrased_question = ""
    for chunk in local_llm.stream(standalone_question_prompt):
        rephrased_question += chunk
    
    return rephrased_question.strip()

def query_vector_store(concept):
    """Retrieve relevant documents from FAISS using optimized HNSW indexing."""
    docs = vector_store.similarity_search(concept, k=10)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

def manage_memory():
    """Manages in-memory chat history."""
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    return "\n".join(memory_chunks[-5:])  # Keep last 5 chunks

def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    """Tokenizes text and splits it into smaller overlapping chunks."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def chat_loop(user_id):
    """Chatbot loop that retrieves chat history based on user ID."""
    print("Chatbot started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot stopped.")
            break

        # Retrieve context from DB and memory
        db_context = get_chat_history(user_id)
        memory_context = manage_memory()

        # Generate standalone question
        rephrased_question = generate_standalone_question(db_context + "\n" + memory_context, user_input)

        # Retrieve relevant FAISS results
        retrieved_context = query_vector_store(rephrased_question)

        # Format final LLM prompt
        final_prompt = prompt_.format(question=rephrased_question, context=db_context + "\n" + retrieved_context + "\n" + memory_context)

        response_text = ""
        for chunk in local_llm.stream(final_prompt):
            print(chunk, end="", flush=True)
            response_text += chunk

        print("\n")

        # Save conversation to memory and database
        memory.save_context({"User": user_input}, {"AI": response_text})

        # Save to FAISS
        vector_store.add_texts([f"User: {user_input}\nBot: {response_text}"], embedding=embeddings, metadatas=[{"timestamp": time.time()}])

        # Save to PostgreSQL
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO chat_history (user_id, user_message, assistant_response, timestamp) VALUES (%s, %s, %s, %s);",
                       (user_id, user_input, response_text, time.time()))
        connection.commit()
        cursor.close()
        connection.close()

if __name__ == "__main__":
    user_id = input("Enter User ID: ").strip()
    chat_loop(user_id)
