import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
from langchain.chains import ConversationChain

# Simulated past messages (You can replace this with actual conversation history)
past_messages = [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "Our refund policy allows returns within 30 days with a receipt."},
    {"role": "user", "content": "Do you offer refunds for digital products?"},
    {"role": "assistant", "content": "No, digital products are non-refundable."}
]

# Initialize environment variables (optional)
load_dotenv()

# Convert messages into LangChain Document objects
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

# Create retrieval chain using FAISS
def create_chain(vector_store):
    model = ChatOpenAI(temperature=0.4, model='gpt-3.5-turbo-1106')

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the past conversation history.
    Context: {context}
    Question: {input}
    """)

    # Use the retrieval chain to get relevant context and generate a response
    document_chain = create_retrieval_chain(vector_store.as_retriever(), model)

    return document_chain

# Run the app
def run_chat_app(query, new_message=None):
    # Get context from past messages
    docs = get_context_from_past_messages(past_messages)
    
    # Create or update the FAISS vector store
    vector_store = create_vector_store(docs)

    if new_message:
        # Update FAISS store with the new message
        update_vector_store(vector_store, new_message)

    # Create retrieval chain for context-aware responses
    chain = create_chain(vector_store)
    
    # Invoke the chain with the user query
    response = chain.invoke({"input": query})
    
    return response

# Example of running the chat app with a new message
query = "What is the refund policy?"
response = run_chat_app(query)
print("Response to query:", response)

# Example of adding a new message and getting a response
new_message = {"role": "user", "content": "Can I return products after 30 days?"}
response_with_new_message = run_chat_app("What is the policy for returns?", new_message=new_message)
print("Response with new message:", response_with_new_message)
