import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# Load environment variables (optional)
load_dotenv()

# Database connection URL (SQLAlchemy format)
DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER', 'your_user')}:{os.getenv('DB_PASSWORD', 'your_password')}@" \
         f"{os.getenv('DB_HOST', 'your_host')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'your_database')}"

# Initialize SQLAlchemy Engine & Session
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

# Retrieve data from PostgreSQL using SQLAlchemy
def get_docs():
    """
    Fetch data from PostgreSQL and convert it into LangChain Documents.
    """
    try:
        with SessionLocal() as session:
            result = session.execute(text("SELECT id, content FROM documents_table;"))  # Adjust query/table name
            rows = result.fetchall()
        
        # Convert to LangChain Documents
        docs = [Document(page_content=row[1], metadata={"id": row[0]}) for row in rows]

        # Split into chunks
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
