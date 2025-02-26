def chat(user_input):
    """Handles standalone questions using stored chat history."""
    
    # Retrieve past messages from DB
    past_messages = get_past_messages()
    
    # Initialize ChatGPT model
    llm = ChatOpenAI(model="gpt-4")

    # Define a prompt template
    chat_prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="Past conversation:\n{history}\nNow answer this standalone question: {question}"
    )

    # Create an LLM chain
    chat_chain = LLMChain(llm=llm, prompt=chat_prompt)

    # Generate a response
    response = chat_chain.run({"history": past_messages, "question": user_input})

    # Save the conversation to the database
    save_message_to_db(user_input, response)
    
    return response
