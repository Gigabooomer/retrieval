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

# Async function to fetch the last 10 added values for a specific ID
async def fetch_last_10_values(specific_id):
    async with AsyncSessionLocal() as session:  # ✅ Correct async session usage
        query = text("""
            SELECT column_name
            FROM table_name
            WHERE id = :id
            ORDER BY created_at DESC
            LIMIT 10;
        """)
        
        result = await session.execute(query, {"id": specific_id})  # ✅ Await async execution
        rows = result.scalars().all()  # Extract results

        return rows

# Run the async function
async def main():
    specific_id = 123  # Example ID
    results = await fetch_last_10_values(specific_id)
    print(results)

# Run in an event loop
asyncio.run(main())
