import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize FAISS vector store from LangChain
vector_store = FAISS.from_texts([], embedding=embeddings)  # Start empty

# Define prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="Chat History:\n{chat_history}\n\nUser: {user_input}\nBot:",
)

# Initialize LangChain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize OpenAI LLM with streaming enabled
llm = OpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Initialize LLMChain with OpenAI model
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

# Function to chunk messages using RecursiveCharacterTextSplitter
def chunk_text(text, chunk_size=256, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to store user input embeddings in FAISS
def store_user_message(user_message):
    chunks = chunk_text(user_message)  
    if not chunks:
        print("Warning: Empty chunks for message:", user_message)
        return
    vector_store.add_texts(chunks)
    memory.save_context({"input": user_message}, {"output": "..."})  # Placeholder

# Function to retrieve relevant past messages from FAISS
def retrieve_past_context(user_message, k=3):
    if len(vector_store.index) == 0:
        return []
    docs = vector_store.similarity_search(user_message, k=k)
    return [doc.page_content for doc in docs] if docs else []

# Function to stream chatbot response
def chat_with_bot(user_input):
    past_context = retrieve_past_context(user_input)
    store_user_message(user_input)

    context_str = "\n".join(past_context)
    full_input = {"chat_history": memory.load_memory_variables({})["chat_history"], "user_input": user_input}

    print("Chatbot: ", end="", flush=True)

    response_text = ""
    for chunk in conversation.stream(full_input):
        chunk_text = chunk.get("text", "")
        response_text += chunk_text
        print(chunk_text, end="", flush=True)  # Print each chunk in real-time

    print("\n")  # Move to a new line after streaming is complete
    memory.save_context({"input": user_input}, {"output": response_text})  # Save response

# Run chatbot loop
def run_chatbot():
    print("Chatbot is running! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        chat_with_bot(user_input)

# Start chatbot
run_chatbot()
