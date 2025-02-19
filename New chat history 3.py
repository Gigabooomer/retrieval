import os
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set OpenAI API key for embeddings
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize OpenAI Embeddings (or replace with a local embedding model)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize FAISS index for storing chat history embeddings
embedding_size = 1536  # OpenAI's ada-002 embedding dimension
index = faiss.IndexFlatL2(embedding_size)
chat_history = []  # Store raw text chat history
vector_store = []  # Store embeddings linked to chat history

# Load a Local LLM Model (Replace with your preferred model)
model_name = "mistralai/Mistral-7B-Instruct"  # Change to your deployed model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipeline_llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Wrap the model in LangChain's LLM interface
local_llm = HuggingFacePipeline(pipeline=pipeline_llm)

# Define a simple prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="Chat History:\n{chat_history}\n\nUser: {user_input}\nBot:",
)

# Initialize LangChain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLMChain with Local Model
conversation = LLMChain(
    llm=local_llm,
    prompt=prompt,
    memory=memory,
)

# Function to store user input embeddings in chat history
def store_user_message(user_message):
    global chat_history, vector_store, index
    
    # Generate embedding
    vector = np.array([embeddings.embed_query(user_message)], dtype=np.float32)
    
    # Store text and embedding
    chat_history.append(user_message)
    vector_store.append(vector)
    index.add(vector)  # Add to FAISS index

# Function to retrieve most relevant past messages
def retrieve_past_context(user_message, k=3):
    if len(vector_store) == 0:
        return []  # No history yet

    query_vector = np.array([embeddings.embed_query(user_message)], dtype=np.float32)
    D, I = index.search(query_vector, k=min(k, len(vector_store)))  # Search history

    # Retrieve matching messages
    retrieved_messages = [chat_history[i] for i in I[0] if i < len(chat_history)]
    return retrieved_messages

# Function to chat with the bot using local LLM + embeddings
def chat_with_bot(user_input):
    # Retrieve relevant past messages using embeddings
    past_context = retrieve_past_context(user_input)
    
    # Store user input in both text memory and embeddings
    memory.save_context({"input": user_input}, {"output": "..."})  # Placeholder
    store_user_message(user_input)

    # Append relevant history for context
    context_str = "\n".join(past_context)
    full_input = f"Context: {context_str}\nUser: {user_input}"
    
    # Get chatbot response
    response = conversation.run(chat_history=memory.load_memory_variables({})["chat_history"], user_input=user_input)
    
    # Store bot response in memory
    memory.save_context({"input": user_input}, {"output": response})
    
    return response

# Run the chatbot in a loop
print("Chatbot is running! Type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    response = chat_with_bot(user_input)
    print("Chatbot:", response)
  
