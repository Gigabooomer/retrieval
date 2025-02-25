import os
from langserve import RemoteRunnable
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# Importing required configurations and functions
from cf import LLM_API
from some_module import run_chain  # Ensure this function is properly defined elsewhere

# Load Hugging Face API token
HF_TOKEN = "hugging_face_token"

# Initialize Remote LLM
local_llm = RemoteRunnable(LLM_API)

# Define file paths
database_schema_file = "path_database_schema"

# Load database schema
with open(database_schema_file, "r") as f:
    database_schema = f.read()

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS Vector Store
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Define Prompt for Standalone Question Generation
standalone_question_prompt = """
You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.

### Chat History:
{chat_history}

### User Input:
{question}

### Standalone Question:
"""

# Function to Tokenize and Chunk Text
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Function to Format and Optimize Chat History (Using Tokenization and Chunking)
def format_chat_history(chat_context, max_turns=10, max_tokens=1024):
    """
    Formats chat history retrieved from the database while ensuring token limits are respected.

    Args:
        chat_context (list): A list of chat records (dictionaries with "user" and "assistant").
        max_turns (int): Maximum turns to consider.
        max_tokens (int): Maximum total tokens allowed for chat history.

    Returns:
        str: Tokenized and optimized chat history.
    """
    recent_context = chat_context[:max_turns]  # Consider only the most recent turns
    formatted_context = "\n".join(
        f"User: {d['user']}\nAssistant: {d['assistant']}" for d in recent_context
    )

    # Tokenize and chunk the chat history to fit within max_tokens limit
    optimized_context = tokenize_and_chunk(formatted_context, max_tokens=max_tokens)

    return "\n".join(optimized_context)

# Function to Generate a Standalone Question
def chat_history(chat_context, question, schema, standalone_question_chain):
    """
    Processes chat history and generates a standalone question.

    Args:
        chat_context (str): The formatted and tokenized chat history.
        question (str): The user's input question.
        schema (str): The database schema.
        standalone_question_chain (Runnable): The LLM chain that reformulates the question.

    Returns:
        str: The standalone question reformulated by the model.
    """
    full_prompt = standalone_question_prompt.format(
        chat_history=chat_context,
        question=question
    )

    # Generate a standalone question using run_chain
    standalone_question = run_chain(
        standalone_question_chain,
        {
            "question": question,
            "context": chat_context,
            "schema": schema
        }
    )

    return standalone_question.strip()
  
