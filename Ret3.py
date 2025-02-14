import asyncio
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL for asyncpg (PostgreSQL)
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER', 'your_user')}:{os.getenv('DB_PASSWORD', 'your_password')}@" \
               f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'your_database')}"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create async session factory
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Async function to fetch **all** values for a specific ID
async def fetch_all_values(specific_id):
    async with AsyncSessionLocal() as session:  # ✅ Correct async session usage
        query = text("""
            SELECT column_name
            FROM table_name
            WHERE id = :id
            ORDER BY created_at DESC;  -- Removed LIMIT to fetch all records
        """)
        
        result = await session.execute(query, {"id": specific_id})  # ✅ Await async execution
        rows = result.scalars().all()  # Extract all results

        return rows

# Run the async function
async def main():
    specific_id = 123  # Example ID
    results = await fetch_all_values(specific_id)
    print(results)  # Print all fetched values

# Run in an event loop
asyncio.run(main())
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://your_user:your_password@127.0.0.1:5432/your_database"

async def test_connection():
    try:
        engine = create_async_engine(DATABASE_URL, echo=True)
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1;")
            print("Database Connected:", result.fetchall())
    except Exception as e:
        print("Connection Failed:", e)

asyncio.run(test_connection())

import asyncio
import asyncpg
import os

async def test_connection():
    try:
        conn = await asyncpg.connect(
            user=os.getenv('DB_USER', 'your_user'),
            password=os.getenv('DB_PASSWORD', 'your_password'),
            database=os.getenv('DB_NAME', 'your_database'),
            host=os.getenv('DB_HOST', '127.0.0.1'),  # Use 127.0.0.1 instead of "localhost"
            port=os.getenv('DB_PORT', '5432')
        )
        print("✅ Connected to database successfully!")
        await conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

asyncio.run(test_connection())

