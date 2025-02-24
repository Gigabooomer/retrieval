import faiss
import numpy as np
from langserve import RemoteRunnable
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from dotenv import load_dotenv

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

# Load the prompt template from an external file
TEMPLATE_FILE = "prompt_template.txt"
with open(TEMPLATE_FILE, "r") as f:
    template_content = f.read().strip()

# Define Prompt Template
prompt_ = PromptTemplate.from_template(template_content)

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

def retrieve_past_context(chat_history, user_message, k=20):
    """Retrieve relevant past context from FAISS and memory, combined with provided chat history."""
    
    # Get FAISS context
    docs = vector_store.similarity_search(user_message, k=k)
    faiss_context = [doc.page_content for doc in docs] if docs else []

    # Convert retrieved chat history into a formatted string
    formatted_history = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])

    # Get memory-stored context
    memory_history = memory.load_memory_variables({}).get("chat_history", [])
    memory_context = [msg.content for msg in memory_history] if memory_history else []

    return "\n".join(faiss_context + [formatted_history] + memory_context)

def generate_standalone_question(chat_history, thread_id, user_input):
    """Rewrites user input into a standalone question based on chat history."""
    
    # Retrieve relevant past context
    past_context = retrieve_past_context(chat_history, user_input)

    # Construct prompt for standalone question generation
    standalone_question_prompt = f"""
    You are an AI that reformulates user questions into standalone queries.
    
    ### Chat History:
    {past_context}

    ### User Input:
    {user_input}

    ### Standalone Question:
    """
    
    rephrased_question = ""
    for chunk in local_llm.stream(standalone_question_prompt):
        rephrased_question += chunk

    return rephrased_question.strip()
