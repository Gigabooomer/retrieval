import os
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Embeddings and FAISS Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
vector_store = FAISS.from_texts([" "], embedding=embeddings)

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
    template = f.read().strip()  # Remove extra whitespace

# Load SQL schema
with open(SCHEMA_FILE, "r") as f:
    schema = f.read().strip()  # Remove extra whitespace

# Define Prompt Template
prompt_ = PromptTemplate.from_template(template)

# Function to chunk text before storing in FAISS
def chunk_text(text, chunk_size=256, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to retrieve relevant past context
def retrieve_past_context(user_message, k=10):
    docs = vector_store.similarity_search(user_message, k=k)
    if not docs:
        print(f"Warning: No relevant past context found for: {user_message}")
        return []
    return [doc.page_content for doc in docs]

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

    # Load memory context
    memory_context = "\n".join(memory.load_memory_variables({}).get("chat_history", []))

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
