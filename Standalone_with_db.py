# Assume Chat_context is a list of dicts like:
# Chat_context = [{"user": "Hello", "assistant": "Hi! How can I help?"}, ...]

def retrieve_db_context():
    return "\n".join([f"User: {entry['user']}\nBot: {entry['assistant']}" for entry in Chat_context])

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    past_context = retrieve_past_context(question)
    db_context = retrieve_db_context()
    memory_context = manage_memory()

    combined_context = db_context + "\n" + "\n".join(past_context) + "\n" + memory_context
    rephrased_question = rephrase_question(combined_context, question)

    final_prompt = prompt_.format(question=rephrased_question, context=combined_context)

    print("\n--- Rephrased Question ---")
    print(rephrased_question)
    print("--------------------\n")

    response_text = ""
    for chunk in local_llm.stream(final_prompt):
        print(chunk, end="", flush=True)
        response_text += chunk

    print("\n")

    memory.save_context({"User": question}, {"AI": response_text})

    vector_store.add_texts(
        [f"User: {question}\nBot: {response_text}"], 
        embedding=embeddings, 
        metadatas=[{"timestamp": time.time()}]
    )
