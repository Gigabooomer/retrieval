# db_connection.py
import psycopg2

# PostgreSQL database configuration
DB_NAME = "llm_benchmark"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def connect_to_db():
    """Establish a connection to the PostgreSQL database and return the connection object."""
    try:
        DB_PASSWORD = input("Enter PostgreSQL password: ")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("✅ Successfully connected to PostgreSQL database!")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None

if __name__ == "__main__":
    connection = connect_to_db()
    if connection:
        connection.close()
        print("✅ Connection closed successfully!")
