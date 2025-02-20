from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
cache_dir = "./models/deepseek_qwen7b"
hf_token = "YOUR_HF_TOKEN_HERE"

# Downloads (or loads from cache) the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir=cache_dir, 
    use_auth_token=hf_token
)

# Downloads (or loads from cache) the model and places it on available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=cache_dir, 
    device_map="auto", 
    use_auth_token=hf_token
)
