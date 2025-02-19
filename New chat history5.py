import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set OpenAI API key for embeddings and LLM
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize FAISS vector store from LangChain
vector_store = FAISS.from_texts([], embedding=embeddings)  # Start empty

# Define a simple prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="Chat History:\n{chat_history}\n\nUser: {user_input}\nBot:",
)

# Initialize LangChain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize OpenAI LLM (GPT-3.5 or GPT-4)
llm = OpenAI(model_name="gpt-3.5-turbo")  # You can change this to GPT-4 if needed

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
    # Split user message into chunks
    chunks = chunk_text(user_message)  
    
    # Check if chunks are empty
    if not chunks:
        print("Warning: Empty chunks for message:", user_message)
        return
    
    # Add chunks to the FAISS vector store
    vector_store.add_texts(chunks)  # Store in FAISS
    memory.save_context({"input": user_message}, {"output": "..."})  # Placeholder

# Function to retrieve relevant past messages from FAISS
def retrieve_past_context(user_message, k=3):
    # Check if vector store has any documents
    if len(vector_store.index) == 0:
        return []  # Return empty if no documents are in FAISS
    
    docs = vector_store.similarity_search(user_message, k=k)
    
    if not docs:
        print("Warning: No relevant past context found for:", user_message)
        return []  # Return empty if no relevant docs are found

    return [doc.page_content for doc in docs]

# Function to process user input
def chat_with_bot(user_input):
    past_context = retrieve_past_context(user_input)

    # Store user input in FAISS
    store_user_message(user_input)

    # Prepare context for the bot
    context_str = "\n".join(past_context)
    full_input = f"Context: {context_str}\nUser: {user_input}"

    # Get chatbot response from OpenAI GPT
    response = conversation.run(chat_history=memory.load_memory_variables({})["chat_history"], user_input=user_input)

    # Store bot response in memory
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
