def chat(user_input):
    if is_chat_based(user_input):  # If contextual, retrieve memory
        memory_context = short_term_memory.load_memory_variables({})["history"]
        memory_context += long_term_memory.load_memory_variables({})["history"]
    else:  # Standalone, no memory needed
        memory_context = ""

    # Generate a response
    chat_prompt = PromptTemplate(
        input_variables=["memory", "question"],
        template="Previous conversation: {memory}\nUser: {question}\nAI:"
    )

    chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chat_chain.run({"memory": memory_context, "question": user_input})
    
    # Store the conversation in memory
    short_term_memory.save_context({"input": user_input}, {"output": response})
    long_term_memory.save_context({"input": user_input}, {"output": response})

    return response
