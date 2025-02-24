import os
import faiss
import numpy as np
import asyncio
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from cn import sessionl
from lgc import get_data_by_id

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
from cf import LLM_API
local_llm = RemoteRunnable(LLM_API)

# Load the database schema from a file
DATABASE_SCHEMA_FILE = "path_database_schema"
with open(DATABASE_SCHEMA_FILE, "r") as f:
    database_schema = f.read().strip()

# Load the prompt template from an external file
TEMPLATE_FILE = "prompt_template.txt"
with open(TEMPLATE_FILE, "r") as f:
    template_content = f.read().strip()

# Define Prompt Template
prompt_ = PromptTemplate.from_template(template_content)

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

async def get_chat_history(user_id):
    """Retrieve chat history from the database using user ID."""
    async with sessionl() as session:
        data = await get_data_by_id(session, user_id=user_id)
        return "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in data]) if data else ""

def retrieve_past_context(user_message, k=20):
    """Retrieve relevant past context from FAISS and memory."""
    docs = vector_store.similarity_search(user_message, k=k)
    faiss_context = [doc.page_content for doc in docs] if docs else []

    memory_history = memory.load_memory_variables({}).get("chat_history", [])
    memory_context = [msg.content for msg in memory_history] if memory_history else []

    return "\n".join(faiss_context + memory_context)

def generate_standalone_question(chat_history, user_input):
    """Rewrites user input into a standalone question."""
    standalone_question_prompt = f"""
    You are an AI that reformulates user questions into standalone queries.
    
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

def chat_loop(user_id, user_input):
    """Processes user input and returns a standalone question."""
    
    # Retrieve context from database
    db_context = asyncio.run(get_chat_history(user_id))

    # Retrieve relevant FAISS results
    retrieved_context = retrieve_past_context(user_input)

    # Generate standalone question
    rephrased_question = generate_standalone_question(db_context + "\n" + retrieved_context, user_input)

    return rephrased_question
