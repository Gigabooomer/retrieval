import os
import time
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage
from transformers import AutoTokenizer

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
    template = f.read().strip()

# Define Prompt Template
prompt_ = PromptTemplate.from_template(template)

# Initialize the tokenizer (Hugging Face tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

# Rephrase prompt template
rephrase_prompt_template = """
You are an AI that reformulates user questions into standalone queries. Given the conversation context and user input, rewrite it as a fully independent question.

### Context:
{context}

### User Input:
{question}

### Standalone Question:
"""

def tokenize_and_chunk(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def retrieve_past_context(user_message, k=15):
    docs = vector_store.similarity_search(user_message, k=k)
    
    if not docs:
        return []
    
    sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get("timestamp", 0), reverse=True)
    
    return [doc.page_content for doc in sorted_docs]

def manage_memory():
    all_memory = memory.load_memory_variables({}).get("chat_history", [])
    
    memory_chunks = []
    for message in all_memory:
        memory_chunks.extend(tokenize_and_chunk(message.content))
    
    return "\n".join(memory_chunks[-5:])  # Keep last 5 chunks

def rephrase_question(context, question):
    full_prompt = rephrase_prompt_template.format(context=context, question=question)
    
    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk
    
    return rephrased_question.strip()

print("Chatbot started. Type 'exit' to stop.")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    past_context = retrieve_past_context(question)
    context_str = "\n".join(past_context)

    memory_context = manage_memory()

    rephrased_question = rephrase_question(context_str + "\n" + memory_context, question)

    final_prompt = prompt_.format(question=rephrased_question, context=context_str + "\n" + memory_context)

    print("\n--- Rephrased Question ---")
    print(rephrased_question)
    print("--------------------\n")

    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")

    memory.save_context({"User": question}, {"AI": response_text})

    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )
