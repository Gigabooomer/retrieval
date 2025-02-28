import spacy
import faiss
import yake
import time
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# Load Spacy NER Model
nlp = spacy.load("en_core_web_sm")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Initialize YAKE for concept extraction
keyword_extractor = yake.KeywordExtractor(n=2, dedupLim=0.9, top=3)

# Initialize Vector Store with FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# Initialize Chat Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------
# 1️⃣ TOKENIZE & CHUNK TEXT
# -------------------------
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# -------------------------
# 2️⃣ RETRIEVE PAST CONTEXT
# -------------------------
def retrieve_past_context(user_message, k=10):
    docs = vector_store.similarity_search(user_message, k=k)
    return "\n".join([doc.page_content for doc in docs]) if docs else ""

# -------------------------------
# 3️⃣ GENERATE STANDALONE QUESTION
# -------------------------------
def generate_standalone_question(chat_history, user_input):
    prompt_template = f"""
    You are an AI that reformulates user questions into standalone queries. Given the conversation history and user input, rewrite it as a fully independent question.
    
    ### Chat History:
    {chat_history}

    ### User Input:
    {user_input}

    ### Standalone Question:
    """
    
    response = ""
    for chunk in local_llm.stream(prompt_template):
        response += chunk
    return response.strip()

# -----------------------------
# 4️⃣ EXTRACT CONCEPT FROM TEXT
# -----------------------------
def extract_concepts(text):
    keywords = keyword_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

# -----------------------------
# 5️⃣ QUERY VECTOR STORE
# -----------------------------
def query_vector_store(concept):
    docs = vector_store.similarity_search(concept, k=10)
    return "\n".join([doc.page_content for doc in docs])

# -----------------------------
# 6️⃣ MANAGE CHAT MEMORY
# -----------------------------
def manage_memory():
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    memory_chunks = []
    for msg in all_memory:
        memory_chunks.extend(tokenize_and_chunk(msg.content))
    return "\n".join(memory_chunks[-10:])  # Keep last 10 messages

# -------------------------
# 7️⃣ MAIN CHATBOT FUNCTION
# -------------------------
def chatbot(user_input):
    # Retrieve past context from vector DB
    past_context = retrieve_past_context(user_input)

    # Manage memory-based context
    memory_context = manage_memory()

    # Generate standalone question
    rephrased_question = generate_standalone_question(memory_context, user_input)

    # Extract key concepts
    concepts = extract_concepts(rephrased_question)

    # Retrieve related context from vector store
    retrieved_context = "\n".join([query_vector_store(concept) for concept in concepts])

    # Final Prompt for LLM
    final_prompt = f"""
    You are an AI assistant helping users with SQL queries and database-related questions.

    ### Past Context:
    {past_context}

    ### Memory Context:
    {memory_context}

    ### Extracted Concepts:
    {', '.join(concepts)}

    ### Reformulated Question:
    {rephrased_question}

    ### Answer:
    """
    
    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        response_text += chunk

    # Save to memory
    memory.save_context({"User": user_input}, {"Assistant": response_text})

    # Add to vector store
    vector_store.add_texts([f"User: {user_input}\nAssistant: {response_text}"], embedding=embeddings, metadatas=[{"timestamp": time.time()}])

    return response_text.strip()

# -------------------------
# USAGE EXAMPLE
# -------------------------
if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        response = chatbot(user_input)
        print("\nAssistant:", response)
