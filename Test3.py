import os
from langchain_core.runnables import RemoteRunnable
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Configuration
LLM_API = "your_llm_endpoint_here"  # Replace with actual LLM API URL
TEMPLATE_FILE = "template.txt"  # Path to the prompt template
SCHEMA_FILE = "schema.txt"  # Path to the schema file

# Load LLM
llm = RemoteRunnable(LLM_API)

# Load prompt template
with open(TEMPLATE_FILE, "r") as f:
    template = f.read()

prompt_ = PromptTemplate.from_template(template)

# Load schema
with open(SCHEMA_FILE, "r") as f:
    schema = f.read()

# Initialize memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

print("🔹 Chatbot running... (Type 'exit' to stop)")

while True:
    question = input("\nEnter question: ").strip()

    if question.lower() == "exit":
        print("🔹 Exiting chatbot. Goodbye!")
        break

    if not question:
        print("⚠️ Please enter a valid question.")
        continue

    # Retrieve history from memory
    history = memory.load_memory_variables({}).get("history", "")

    # Ensure history is formatted properly
    if isinstance(history, list):
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # Construct a standalone question
    standalone_prompt = f"""Rewrite the following question into a self-contained question, considering past conversation context:
    
    History:
    {history}
    
    User's Question: {question}
    """

    # Generate self-contained question
    standalone_question = llm.invoke({"input": standalone_prompt}).strip()
    print(f"\n🔄 Rewritten Question: {standalone_question}")

    # Prepare final prompt for the LLM
    final_prompt = prompt_.format(schema=schema, question=standalone_question)

    # Generate AI response
    response = ""
    print("\n🤖 AI Response:", end=" ")

    for chunk in llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response += chunk
    print("\n")  # Ensure proper formatting after response

    # Save conversation history
    memory.save_context({"input": standalone_question}, {"output": response})
