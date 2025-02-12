import os
import psycopg2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# Load environment variables (optional)
load_dotenv()

# Database connection configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "your_database"),
    "user": os.getenv("DB_USER", "your_user"),
    "password": os.getenv("DB_PASSWORD", "your_password"),
    "host": os.getenv("DB_HOST", "your_host"),
    "port": os.getenv("DB_PORT", "your_port"),
}

# Retrieve data from PostgreSQL
def get_docs():
    """
    Fetch data from the PostgreSQL database and convert it into LangChain Documents.
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, content FROM documents_table;")  # Adjust query/table name
                rows = cursor.fetchall()
        
        # Convert to LangChain Documents
        docs = [Document(page_content=row[1], metadata={"id": row[0]}) for row in rows]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splitDocs = text_splitter.split_documents(docs)

        return splitDocs
    
    except Exception as e:
        print(f"Error fetching documents from DB: {e}")
        return []

# Create vector store
def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

# Create retrieval chain
def create_chain(vectorStore):
    model = ChatOpenAI(temperature=0.4, model='gpt-3.5-turbo-1106')

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the database records.
    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Run the pipeline
docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke({"input": "What is the policy for refunds?"})
print(response)
