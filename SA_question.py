from sqlalchemy import create_engine, text
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# PostgreSQL connection
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/chat_db"
engine = create_engine(DATABASE_URL)

def get_last_question(session_id):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT user_query FROM chat_history WHERE session_id = :session_id ORDER BY turn_id DESC LIMIT 1"),
            {"session_id": session_id}
        ).fetchone()
        
        return result[0] if result else None

def generate_standalone_question(question):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Rewrite the following question as a standalone question without relying on context:\n\n{question}"
    )
    
    llm = OpenAI(model_name="gpt-4")
    return llm(prompt.format(question=question))

session_id = "session_123"
last_question = get_last_question(session_id)

if last_question:
    standalone_question = generate_standalone_question(last_question)
    print(standalone_question)
