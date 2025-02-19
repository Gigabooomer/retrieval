import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set OpenAI API key for embeddings
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize OpenAI Embeddings (or replace with a local embedding model)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize FAISS vector store from LangChain
vector_store = FAISS.from_texts([], embedding=embeddings)  # Start empty

# Load a Local LLM Model (Replace with your preferred model)
def load_local_llm(model_name="mistralai/Mistral-7B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipeline_llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipeline_llm)

local_llm = load_local_llm()

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

# Function to chunk messages using RecursiveCharacterTextSplitter
def chunk_text(text, chunk_size=256, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to store user input embeddings in FAISS
def store_user_message(user_message):
    chunks = chunk_text(user_message)  # Split message into chunks
    vector_store.add_texts(chunks)  # Store in FAISS
    memory.save_context({"input": user_message}, {"output": "..."})  # Placeholder

# Function to retrieve relevant past messages from FAISS
def retrieve_past_context(user_message, k=3):
    docs = vector_store.similarity_search(user_message, k=k)
    return [doc.page_content for doc in docs]

# Function to process user input
def chat_with_bot(user_input):
    past_context = retrieve_past_context(user_input)

    # Store user input
    store_user_message(user_input)

    # Prepare context
    context_str = "\n".join(past_context)
    full_input = f"Context: {context_str}\nUser: {user_input}"

    # Get chatbot response
    response = conversation.run(chat_history=memory.load_memory_variables({})["chat_history"], user_input=user_input)

    # Store bot response
    memory.save_context({"input": user_input}, {"output": response})

    return response

# Run the chatbot in a loop
def run_chatbot():
    print("Chatbot is running! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = chat_with_bot(user_input)
        print("Chatbot:", response)

# Start chatbot
run_chatbot()
