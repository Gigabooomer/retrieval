import os
import asyncio
from langserve import RemoteRunnable
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from cf import LLM_API

# Initialize Remote LLM API
local_llm = RemoteRunnable(LLM_API)

# HuggingFace Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Define the Prompt Template
standalone_question_prompt = """
You are an AI that reformulates user questions into standalone queries. 
Given the conversation history and user input, rewrite it as a fully independent question.

### Chat History:
{chat_history}

### User Input:
{question}

### Standalone Question:
"""

# Function to tokenize and chunk text
def tokenize_and_chunk(text, max_tokens=768, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Function to format the retrieved database chat history
def format_chat_history(chat_context, max_turns=10):
    """Formats chat history passed from the database to maintain coherence."""
    recent_context = chat_context[:max_turns]
    return "\n".join([f"User: {d['user']}\nAssistant: {d['assistant']}" for d in recent_context])

# Function to generate a standalone question
def generate_standalone_question(chat_history, user_input):
    """Generates a standalone question based on the provided chat history."""
    full_prompt = standalone_question_prompt.format(chat_history=chat_history, question=user_input)
    
    rephrased_question = ""
    for chunk in local_llm.stream(full_prompt):
        rephrased_question += chunk

    return rephrased_question.strip()

# Main function to process the input and return a standalone question
def process_chat_history(chat_context, user_input):
    """Takes database chat context and user input, returns a standalone question."""
    formatted_chat_history = format_chat_history(chat_context)
    return generate_standalone_question(formatted_chat_history, user_input)
