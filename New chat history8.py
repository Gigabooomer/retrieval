from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ChatMemory:
    def __init__(self, embedding_model="text-embedding-ada-002", chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = FAISS.from_texts([], self.embeddings)  # Empty FAISS store initially

    def add_to_memory(self, user_input):
        """Chunk, embed, and store user input"""
        chunks = self.text_splitter.split_text(user_input)
        self.vectorstore.add_texts(chunks)

    def retrieve_relevant_context(self, query, k=3):
        """Retrieve the top-k relevant past inputs"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

class LangChainChatbot:
    def __init__(self):
        self.memory = ChatMemory()
        self.llm = OpenAI(model_name="gpt-3.5-turbo")  # Change as needed

        # Define a prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Context:\n{context}\n\nUser: {query}\nAI:"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def chat(self, user_input):
        """Process input, retrieve context, and generate response"""
        context = self.memory.retrieve_relevant_context(user_input)
        response = self.chain.run(context=context, query=user_input)
        self.memory.add_to_memory(user_input)
        return response

# Example Usage
if __name__ == "__main__":
    bot = LangChainChatbot()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = bot.chat(user_input)
        print(f"AI: {response}")
