def get_last_qa(session_id):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT user_query, assistant_response FROM chat_history WHERE session_id = :session_id ORDER BY turn_id DESC LIMIT 1"),
            {"session_id": session_id}
        ).fetchone()
        
        return result if result else (None, None)

def generate_standalone_question_with_context(question, answer):
    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Rewrite the user's question as a standalone query, incorporating relevant details from the assistant's answer:\n\nUser Question: {question}\nAssistant Response: {answer}\n\nRewritten Standalone Question:"
    )
    
    llm = OpenAI(model_name="gpt-4")
    return llm(prompt.format(question=question, answer=answer))

session_id = "session_123"
question, answer = get_last_qa(session_id)

if question and answer:
    standalone_question = generate_standalone_question_with_context(question, answer)
    print(standalone_question)
