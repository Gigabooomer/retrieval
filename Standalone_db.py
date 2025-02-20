import os
import time
import faiss
import spacy
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

# ✅ Load spaCy NER Model
nlp = spacy.load("en_core_web_sm")

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

# ✅ Initialize FAISS with HNSW Index
d = 3072  
index = faiss.IndexHNSWFlat(d, 32)  
vector_store = FAISS(embedding_function=embeddings, index=index)

# ✅ Define memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Remote LLM API
from config import LLM_2_API
local_llm = RemoteRunnable(LLM_2_API)

# ✅ Paths for template and schema files
TEMPLATE_FILE = "temp_path"
SCHEMA_FILE = "context.sql"

# ✅ Load template
with open(TEMPLATE_FILE, "r") as f:
    template = f.read().strip()

# ✅ Load fixed database schema
with open(SCHEMA_FILE, "r") as f:
    database_schema = f.read().strip()

# ✅ Define Prompt Template
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

# ✅ Sample Context Variable (Simulating Database Data)
database_data = [
    {"message": "What is FAISS?", "response": "FAISS is a library for efficient similarity search."},
    {"message": "How do I improve retrieval in FAISS?", "response": "Use HNSW indexing for better performance."}
]

# ✅ Convert database data into structured format
chat_context = [{"user": d["message"], "assistant": d["response"]} for d in database_data]

# ✅ Step 1: Retrieve Context from Variable (`chat_context`)
def retrieve_context_from_variable():
    """Extract recent conversation history from `chat_context` variable."""
    return "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_context])

# ✅ Step 2: Generate Standalone Question
def generate_standalone_question(chat_history, user_input):
    """Rewrites user input into a standalone question"""
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

# ✅ Step 3: Extract Concept from Standalone Question
def extract_concept_from_question(standalone_question):
    """Extracts key concepts using Named Entity Recognition (NER)."""
    doc = nlp(standalone_question)
    
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "EVENT"]]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    return ", ".join(set(entities + noun_chunks)) if (entities or noun_chunks) else standalone_question

# ✅ Step 4: Query FAISS Vector Store
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
    return "\n".join(memory_chunks[-5:])  

# ✅ Chatbot Loop
print("Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Step 1: Retrieve Context from Variable
    variable_context = retrieve_context_from_variable()

    # Step 2: Manage Memory Context
    memory_context = manage_memory()

    # Step 3: Generate Standalone Question
    rephrased_question = generate_standalone_question(variable_context + "\n" + memory_context, question)

    # Step 4: Extract Key Concepts
    extracted_concept = extract_concept_from_question(rephrased_question)

    # Step 5: Query FAISS Vector Store
    retrieved_context = query_vector_store(extracted_concept)

    # Step 6: Format Final Prompt (Includes Context from Variable)
    final_prompt = prompt_.format(question=rephrased_question, context=variable_context + "\n" + retrieved_context + "\n" + memory_context)

    # Debugging: Print the rephrased question before sending to LLM
    print("\n--- Rephrased Question ---")
    print(rephrased_question)
    print("--------------------\n")

    # Step 7: Generate Response from LLM
    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")  

    # Step 8: Save Conversation to Memory
    memory.save_context({"User": question}, {"AI": response_text})

    # Step 9: Store in FAISS for Future Retrieval
    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )
