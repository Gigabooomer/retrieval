import os
import time
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

# ✅ Initialize FAISS Vector Store
vector_store = FAISS.from_texts([" "], embedding=embeddings)

# ✅ Remote LLM API (GPT-4)
from config import LLM_2_API
local_llm = RemoteRunnable(LLM_2_API)

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

main_query_prompt = """
You are an AI assistant. Use the provided context and retrieved documents to generate an accurate response.

### Standalone Question:
{standalone_question}

### Retrieved Context:
{retrieved_context}

### Answer:
"""

# ✅ Step 1: Receive User Question + Chat History
def get_chat_history():
    """Fetch recent chat interactions from memory."""
    return "\n".join([f"User: {entry['user']}\nBot: {entry['assistant']}" for entry in Chat_context])

# ✅ Step 2: Generate Standalone Question
def generate_standalone_question(chat_history, user_input):
    full_prompt = standalone_question_prompt.format(chat_history=chat_history, question=user_input)
    
    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk
    
    return rephrased_question.strip()

# ✅ Step 3: Get Embedding for Standalone Question
def get_embedding(question):
    return embeddings.embed_query(question)

# ✅ Step 4: Extract Concept from Standalone Question
def solrAI_extractConceptFromQuestion(standalone_question):
    """Simulated function to extract concept."""
    return standalone_question  # Placeholder for an actual concept extraction logic

# ✅ Step 5: Query Vector Store with Concept
def solrAI_getContext(standalone_question):
    """Retrieve relevant documents from FAISS based on concept."""
    docs = vector_store.similarity_search(standalone_question, k=10)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# ✅ Step 6: Retrieve Query Result Documents (Handled in FAISS)
def retrieve_documents(standalone_question):
    return solrAI_getContext(standalone_question)

# ✅ Step 7: Submit to GPT-4 for Final Answer
def generate_final_answer(standalone_question, retrieved_context):
    full_prompt = main_query_prompt.format(standalone_question=standalone_question, retrieved_context=retrieved_context)

    response_text = ""
    for chunk in local_llm.stream(full_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")
    return response_text.strip()

# ✅ Step 8: Display Answer
def chatbot():
    print("Chatbot started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot stopped.")
            break

        # Step 1: Fetch Chat History
        chat_history = get_chat_history()

        # Step 2: Generate Standalone Question
        standalone_question = generate_standalone_question(chat_history, user_input)

        # Step 3: Embed Standalone Question
        embedding = get_embedding(standalone_question)

        # Step 4: Extract Concept
        concept = solrAI_extractConceptFromQuestion(standalone_question)

        # Step 5: Query Vector Store
        retrieved_context = solrAI_getContext(concept)

        # Step 6: Retrieve Documents
        documents = retrieve_documents(standalone_question)

        # Step 7: Submit Standalone Question + Context to GPT-4
        final_answer = generate_final_answer(standalone_question, retrieved_context)

        # Step 8: Display Answer
        print("\nBot:", final_answer)

        # ✅ Store User Input + Response in Chat Context & FAISS
        Chat_context.insert(0, {"user": user_input, "assistant": final_answer})
        vector_store.add_texts([f"User: {user_input}\nBot: {final_answer}"], embedding=embeddings, metadatas=[{"timestamp": time.time()}])

# Run chatbot
chatbot()
