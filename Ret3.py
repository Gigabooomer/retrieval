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

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Construct DATABASE_URL using f-string
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}" \
               f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

print(DATABASE_URL)  # Just for verification (remove in production!)
import os
import urllib.parse
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    DB_NAME: str = Field(..., min_length=1)
    DB_USER: str = Field(..., min_length=1)
    DB_PASSWORD: str = Field(..., min_length=1)
    DB_HOST: str = Field(..., min_length=1)
    DB_PORT: int = Field(..., gt=0, lt=65536)  # Valid port range

    def get_database_url(self) -> str:
        """Returns a properly formatted DATABASE_URL with an encoded password."""
        encoded_password = urllib.parse.quote_plus(self.DB_PASSWORD)  # Encode special characters
        return f"postgresql+asyncpg://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

try:
    # Load values from environment variables and validate
    db_config = DatabaseConfig(
        DB_NAME=os.getenv("DB_NAME"),
        DB_USER=os.getenv("DB_USER"),
        DB_PASSWORD=os.getenv("DB_PASSWORD"),
        DB_HOST=os.getenv("DB_HOST"),
        DB_PORT=int(os.getenv("DB_PORT", 5432)),  # Default to 5432 if not set
    )

    # Get the safe database URL
    DATABASE_URL = db_config.get_database_url()
    print("✅ Valid DATABASE_URL:", DATABASE_URL)

except ValidationError as e:
    print("❌ Configuration Error:", e)
