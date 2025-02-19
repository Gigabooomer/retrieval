from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# Initialize components
embedding_model = "text-embedding-ada-002"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = OpenAIEmbeddings(model=embedding_model)
vectorstore = FAISS.from_texts([], embeddings)  # Empty FAISS store initially
llm = OpenAI(model_name="gpt-3.5-turbo")  # Change as needed

def add_to_memory(user_input):
    """Chunk, embed, and store user input."""
    chunks = text_splitter.split_text(user_input)
    vectorstore.add_texts(chunks)

def retrieve_relevant_context(query, k=3):
    """Retrieve top-k relevant past inputs."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def chat(user_input):
    """Retrieve context and generate an AI response."""
    context = retrieve_relevant_context(user_input)
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nAI:"
    response = llm.predict(prompt)
    add_to_memory(user_input)
    return response

# Chat loop
if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot stopped.")
            break
        response = chat(user_input)
        print(f"AI: {response}")
