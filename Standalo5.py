import os
import time
import faiss
import spacy  # ✅ For Concept Extraction
import numpy as np
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ Load spaCy NER Model for Concept Extraction
nlp = spacy.load("en_core_web_sm")  # You can replace with a larger model if needed

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

# ✅ Initialize FAISS with HNSW Index for Faster Retrieval
d = 3072  # Embedding dimension (depends on embedding model output)
index = faiss.IndexHNSWFlat(d, 32)  # HNSW for efficient search
vector_store = FAISS(embedding_function=embeddings, index=index)

# ✅ Define memory for conversation history (Persistent)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Remote LLM API (GPT-4)
from config import LLM_2_API
local_llm = RemoteRunnable(LLM_2_API)

# ✅ Paths for template and schema files
TEMPLATE_FILE = "temp_path"
SCHEMA_FILE = "context.sql"

# ✅ Load template
with open(TEMPLATE_FILE, "r") as f:
    template = f.read().strip()

# ✅ Load fixed database schema (will be included in all queries)
with open(SCHEMA_FILE, "r") as f:
    database_schema = f.read().strip()

# ✅ Define Prompt Template (Now includes schema)
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

# ✅ Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

# ✅ Prompt Templates
standalone_question_prompt = """
You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.

### Chat History:
{chat_history}

### User Input:
{question}

### Standalone Question:
"""

# ✅ Function to tokenize and chunk text
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# ✅ Step 1: Retrieve Past Conversation Context (Optimized)
def retrieve_past_context(user_message, k=20):
    """Retrieve past conversation history from FAISS & Memory"""
    docs = vector_store.similarity_search(user_message, k=k)
    faiss_context = [doc.page_content for doc in docs] if docs else []

    memory_history = memory.load_memory_variables({}).get("chat_history", [])
    memory_context = [msg.content for msg in memory_history] if memory_history else []

    combined_context = faiss_context + memory_context

    return "\n".join(combined_context) if combined_context else "No relevant past context found."

# ✅ Step 2: Generate Standalone Question
def generate_standalone_question(chat_history, user_input):
    """Rewrites user input into a standalone question"""
    full_prompt = standalone_question_prompt.format(chat_history=chat_history, question=user_input)
    
    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk
    
    return rephrased_question.strip()

# ✅ Step 3: Extract Concept from Standalone Question
def extract_concept_from_question(standalone_question):
    """Extracts key concepts using Named Entity Recognition (NER)."""
    doc = nlp(standalone_question)
    
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "EVENT"]]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    concepts = list(set(entities + noun_chunks))
    
    return ", ".join(concepts) if concepts else standalone_question

# ✅ Step 4: Query Vector Store with Concept (FAISS HNSW)
def query_vector_store(concept):
    """Retrieve relevant documents from FAISS using optimized HNSW indexing."""
    docs = vector_store.similarity_search(concept, k=10)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# ✅ Step 5: Manage Memory (Keep Only Recent Context)
def manage_memory():
    """Manages in-memory chat history"""
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    return "\n".join(memory_chunks[-5:])  # Keep last 5 chunks

# ✅ Chatbot Loop
print("Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Step 1: Retrieve Past Conversation Context (Memory + FAISS)
    past_context = retrieve_past_context(question)

    # Step 2: Manage Memory Context
    memory_context = manage_memory()

    # Step 3: Generate Standalone Question
    rephrased_question = generate_standalone_question(past_context, question)

    # Step 4: Extract Key Concepts
    extracted_concept = extract_concept_from_question(rephrased_question)

    # Step 5: Query Vector Store with Concept
    retrieved_context = query_vector_store(extracted_concept)

    # Step 6: Format Final Prompt (Includes Database Schema)
    final_prompt = prompt_.format(question=rephrased_question, context=retrieved_context + "\n" + memory_context)

    # Debugging: Print the rephrased question before sending to LLM
    print("\n--- Rephrased Question ---")
    print(rephrased_question)
    print("--------------------\n")

    # Step 7: Generate Response from LLM
    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")  # Newline for better readability

    # Step 8: Save Conversation to Memory
    memory.save_context({"User": question}, {"AI": response_text})

    # Step 9: Store in FAISS for Future Retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]  # Store with a timestamp
    )
