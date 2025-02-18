from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.runnables import RemoteRunnable
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# Initialize environment variables (optional)
load_dotenv()

# Simulated past messages (initially empty)
past_messages = [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "Our refund policy allows returns within 30 days with a receipt."},
    {"role": "user", "content": "Do you offer refunds for digital products?"},
    {"role": "assistant", "content": "No, digital products are non-refundable."}
]

# Convert past messages into LangChain Document objects
def get_context_from_past_messages(past_messages):
    """
    Converts past chat messages into LangChain Document format.
    """
    docs = [
        Document(page_content=msg["content"], metadata={"role": msg["role"]})
        for msg in past_messages
    ]
    
    # Split into chunks if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

# Create FAISS Vector Store from past messages
def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    return vector_store

# Function to update FAISS vector store with new message
def update_vector_store(vector_store, new_message):
    """
    Updates the FAISS vector store with the latest message.
    """
    # Convert the new message into a LangChain Document
    doc = Document(page_content=new_message["content"], metadata={"role": new_message["role"]})
    
    # Split the document into chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_doc = text_splitter.split_documents([doc])
    
    # Add the new document chunks to FAISS
    vector_store.add_documents(split_doc)

# Main function for continuous conversation
def chat():
    # Get context from past messages and create the FAISS vector store
    docs = get_context_from_past_messages(past_messages)
    vector_store = create_vector_store(docs)
    
    # Create a RemoteRunnable to handle response generation
    model = ChatOpenAI(temperature=0.4, model='gpt-3.5-turbo-1106')
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the past conversation history.
    Context: {context}
    Question: {input}
    """)

    remote_chain = RemoteRunnable(model)

    print("Chatbot: Hello! How can I assist you today?")

    while True:
        # Get the user input (query)
        query = input("You: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        # Add the user message to the past messages
        new_message = {"role": "user", "content": query}
        past_messages.append(new_message)

        # Update FAISS vector store with the new message
        update_vector_store(vector_store, new_message)

        # Prepare the context from the FAISS store
        retriever = vector_store.as_retriever()
        context = retriever.retrieve(query)

        # Use the RemoteRunnable to get a response with context
        response = remote_chain.invoke({"input": query, "context": context})
        
        print(f"Chatbot: {response}")
        
        # Add assistant response to past messages for context continuity
        past_messages.append({"role": "assistant", "content": response})

# Run the chat application
chat()
