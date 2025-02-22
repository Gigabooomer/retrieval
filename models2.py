# models2.py
from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere
from langchain.llms import Anthropic

# Load API-based models
MODELS = {
    "openai-gpt4": ChatOpenAI(model_name="gpt-4", temperature=0),
    "openai-gpt3.5": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    "cohere-command-r": Cohere(model="command-r"),
    "anthropic-claude3": Anthropic(model="claude-3-opus-20240229"),
}

def get_model(model_name):
    """Returns the selected API-based model."""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found! Available models: {list(MODELS.keys())}")
    return MODELS[model_name]

if __name__ == "__main__":
    # Quick test to check available models
    print("Available API-based models:", list(MODELS.keys()))
