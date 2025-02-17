import os
from langchain_core.runnables import RemoteRunnable
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Configuration
LLM_API = "your_llm_endpoint_here"  # Replace with actual LLM API
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

# Use LangChain's memory to store conversation history
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create LLM chain with memory
chain = LLMChain(llm=llm, prompt=prompt_, memory=memory)

print("🔹 Chatbot running... (Type 'exit' to stop)")

while True:
    question = input("Enter question: ").strip()

    if question.lower() == "exit":
        print("🔹 Exiting chatbot. Goodbye!")
        break

    if not question:
        print("⚠️ Please enter a valid question.")
        continue

    # Generate a standalone question (using memory)
    standalone_prompt = (
        "Rewrite the following question into a self-contained question, considering past conversation context:\n"
        f"History:\n{memory.load_memory_variables({})['history']}\n"
        f"User's Question: {question}"
    )

    # Generate self-contained question
    standalone_question = llm.invoke(standalone_prompt)
    print(f"🔄 Rewriting question: {standalone_question}")

    # Prepare final prompt
    prompt = prompt_.format(schema=schema, question=standalone_question)

    # Generate response
    response = ""
    print("🤖 AI Response:")
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
        response += chunk
    print()

    # Store in memory
    memory.save_context({"input": standalone_question}, {"output": response})
