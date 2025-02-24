echo 'import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_NAME = "deepseek-ai/deepseek-llm-7b"

print(" Loading DeepSeek model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

app = FastAPI()

class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
def generate_text(request: RequestData):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=request.max_tokens)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
' > deepseek_server.py
