def get_chat_history(session_id, limit=5):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT user_query, assistant_response FROM chat_history WHERE session_id = :session_id ORDER BY turn_id DESC LIMIT :limit"),
            {"session_id": session_id, "limit": limit}
        ).fetchall()
    
    # Reverse order to maintain chat flow
    return result[::-1]

def generate_response_with_chat_history(chat_history, new_question):
    chat_prompt = "Here is the chat history:\n"
    
    for q, a in chat_history:
        chat_prompt += f"User: {q}\nAssistant: {a}\n"
    
    chat_prompt += f"\nUser: {new_question}\nAssistant:"
    
    llm = OpenAI(model_name="gpt-4")
    return llm(chat_prompt)

session_id = "session_123"
chat_history = get_chat_history(session_id)
new_question = "How does this apply to real-world cases?"  # Example

response = generate_response_with_chat_history(chat_history, new_question)
print(response)
