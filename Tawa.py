import os
import time
import asyncio
from langserve import RemoteRunnable
from langchain.memory import ConversationBufferMemory
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

# User ID (to be passed from the main file)
user_id = "user_id"

# Path to the database schema file
database_schema_file = "path_database_schema"

# Load database schema
with open(database_schema_file, "r") as f:
    database_schema = f.read()

# Async function to fetch chat history from the database
async def get_h():
    async with sessionl() as session:
        data = await get_data_by_id(session, user_id=user_id)
        
        # Convert retrieved data to the expected format
        data_return = [{"user": d["message"], "assistant": d["response"]} for d in data]
        
        return data_return

# Retrieve database chat history
data_return = asyncio.run(get_h())

# Format database history for context
data_formatted = "\n".join(
    [f"User: {d['user']}\nAssistant: {d['assistant']}" for d in data_return]
)

# Define prompt template
prompt_ = PromptTemplate.from_template(
    f"""
    System: You are an AI assistant helping users with SQL queries and database-related questions.

    ### Database Schema:
    {database_schema}

    ### Database History:
    {data_formatted}

    ### Context:
    {{context}}

    ### User Question:
    {{question}}

    ### Answer:
    """
)

# Standalone question transformation template
standalone_question_prompt = """
System: You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.

### Chat History:
{chat_history}

### User Input:
{question}

### Standalone Question:
"""

# Initialize embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Function to tokenize and chunk text
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Function to retrieve past conversation context from FAISS and memory
def retrieve_past_context(user_message, k=20):
    docs = vector_store.similarity_search(user_message, k=k)
    faiss_context = [doc.page_content for doc in docs] if docs else []

    memory_history = memory.load_memory_variables({}).get("chat_history", [])
    memory_context = [msg.content for msg in memory_history] if memory_history else []

    combined_context = faiss_context + memory_context
    return "\n".join(combined_context)

# Function to generate a standalone question
def generate_standalone_question(chat_history, user_input):
    full_prompt = standalone_question_prompt.format(
        chat_history=chat_history, question=user_input
    )

    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk

    return rephrased_question.strip()

# Function to query FAISS for relevant context
def query_vector_store(concept):
    docs = vector_store.similarity_search(concept, k=20)
    return "\n".join([doc.page_content for doc in docs])

# Function to manage conversation memory
def manage_memory():
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    return "\n".join(memory_chunks[-10:])

# Chatbot loop
print("Chatbot started.")
while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve past conversation context
    past_context = retrieve_past_context(question)

    # Manage memory
    memory_context = manage_memory()

    # Generate a standalone question
    rephrased_question = generate_standalone_question(past_context, question)

    # Query vector store for context
    retrieved_context = query_vector_store(rephrased_question)

    # Format final prompt
    final_prompt = prompt_.format(
        question=rephrased_question, context=retrieved_context + "\n" + memory_context
    )

    # Print the rephrased question
    print("\n--- Rephrased Question ---")
    print(rephrased_question)
    print("--------------------\n")

    # Generate response from LLM
    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")

    # Save conversation to memory
    memory.save_context({"User": question}, {"Assistant": response_text})

    # Store user input and response in FAISS
    vector_store.add_texts(
        [f"User: {question}\nAssistant: {response_text}"],
        embedding=embeddings,
        metadatas=[{"timestamp": time.time()}]
    )
