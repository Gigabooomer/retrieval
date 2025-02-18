from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# Simulated past messages
past_messages = [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "Our refund policy allows returns within 30 days with a receipt."},
    {"role": "user", "content": "Do you offer refunds for digital products?"},
    {"role": "assistant", "content": "No, digital products are non-refundable."}
]

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

# Create retrieval chain using FAISS
def create_chain(vector_store):
    model = ChatOpenAI(temperature=0.4, model='gpt-3.5-turbo-1106')

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the past conversation history.
    Context: {context}
    Question: {input}
    """)

    document_chain = create_retrieval_chain(vector_store.as_retriever(), model)

    return document_chain

# Process messages and create FAISS vector store
docs = get_context_from_past_messages(past_messages)
vector_store = create_vector_store(docs)
chain = create_chain(vector_store)

# Run the pipeline
query = "What is the policy for refunds?"
response = chain.invoke({"input": query})
print(response)
