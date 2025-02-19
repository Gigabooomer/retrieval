import os
import time
import faiss
import numpy as np
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi  # BM25 for keyword retrieval

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Embeddings with a high-quality model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

# Initialize FAISS with HNSW index for faster search
d = 3072  # Embedding dimension (check your model's output size)
index = faiss.IndexHNSWFlat(d, 32)  # HNSW for fast retrieval
vector_store = FAISS(embedding_function=embeddings, index=index)

# BM25 Keyword Search Setup
bm25_corpus = []  # Store past messages
bm25_tokenizer = lambda text: text.lower().split()
bm25_index = None  # Will be initialized after first message

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

# Initialize the tokenizer (Hugging Face GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

# Function to tokenize and chunk text more efficiently
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Dynamic `k` based on query length
def dynamic_k(query):
    length = len(query.split())
    if length < 10:
        return 10
    elif 10 <= length <= 30:
        return 20
    return 30  # Long queries require more recall

# Retrieve past context using FAISS + BM25 Hybrid RAG
def retrieve_past_context(user_message):
    k = dynamic_k(user_message)

    # FAISS similarity search
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(user_message, k=k)
    filtered_docs = [doc for doc, score in docs_with_scores if score >= 0.7]

    # BM25 Keyword Search
    keyword_hits = []
    if bm25_index:
        tokenized_query = bm25_tokenizer(user_message)
        scores = bm25_index.get_scores(tokenized_query)
        ranked_docs = sorted(zip(bm25_corpus, scores), key=lambda x: x[1], reverse=True)
        keyword_hits = [doc[0] for doc in ranked_docs[:5]]  # Top 5 matches

    # Merge FAISS & BM25 results
    combined_results = list(set(filtered_docs + keyword_hits))

    return [doc.page_content for doc in combined_results] if combined_results else []

# Manual memory management: Store only recent and relevant conversation history
def manage_memory():
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    return "\n".join(memory_chunks[-5:])  # Keep last 5 chunks

# Chatbot loop
print("Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve past conversation context
    past_context = retrieve_past_context(question)
    context_str = "\n".join(past_context)

    # Load and manage memory context
    memory_context = manage_memory()

    # Format final prompt
    prompt = prompt_.format(question=question, context=context_str + "\n" + memory_context)

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

    # Store user input and response in FAISS for future retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )

    # Update BM25 corpus and index
    bm25_corpus.append(f"User: {question}\nBot: {response_text}")
    bm25_index = BM25Okapi([bm25_tokenizer(doc) for doc in bm25_corpus]) if bm25_corpus else None
