import os
import asyncio
from langserve import RemoteRunnable
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from cf import LLM_API
from cn import sessionl
from lgc import get_data_by_id

# Hugging Face API Token
HF_TOKEN = "hugging_face_token"

# Remote LLM API
local_llm = RemoteRunnable(LLM_API)

# Path to the database schema file
database_schema_file = "path_database_schema"

# Load database schema
with open(database_schema_file, "r") as f:
    database_schema = f.read()

# Initialize embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Function to tokenize and chunk text
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Async function to fetch chat history from the database
async def get_chat_history(thread_id):
    async with sessionl() as session:
        data = await get_data_by_id(session, thread_id=thread_id)
        
        # Convert retrieved data to expected format
        data_return = [{"user": d["message"], "assistant": d["response"]} for d in data]
        return data_return

# Define prompt templates
standalone_question_prompt = PromptTemplate.from_template("""
System: You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.

### Chat History:
{chat_history}

### User Input:
{question}

### Standalone Question:
""")

main_prompt_template = PromptTemplate.from_template("""
You are an AI assistant helping users with SQL queries and database-related questions.

### Database Schema:
{database_schema}

### Database history
{chat_history}

### Context:
{context}

### User Question:
{question}

### Answer:
""")

# Function to generate a standalone question
def generate_standalone_question(chat_history, user_input):
    full_prompt = standalone_question_prompt.format(
        chat_history=chat_history, question=user_input
    )

    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk

    return rephrased_question.strip()

# Function to generate the final prompt with history, schema, and context
def generate_final_prompt(chat_history, standalone_question, context=""):
    return main_prompt_template.format(
        database_schema=database_schema,
        chat_history=chat_history,
        context=context,
        question=standalone_question
    )

# Main callable function to return a processed standalone question
async def process_chat_history(thread_id, user_input):
    # Fetch chat history from the database
    chat_data = await get_chat_history(thread_id)

    # Format chat history
    chat_history = "\n".join(
        [f"User: {d['user']}\nAssistant: {d['assistant']}" for d in chat_data]
    )

    # Generate standalone question
    standalone_question = generate_standalone_question(chat_history, user_input)

    return standalone_question
