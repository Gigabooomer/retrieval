import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Optionally, you can set your token as an environment variable
# HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def get_config(base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", token=None):
    config = AutoConfig.from_pretrained(base_model, use_auth_token=token)
    return config

def get_tokenizer(
    pretrained_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    token=None,
    local_files_only=False,
    resume_download=True,
    padding_side="left",
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        use_auth_token=token,
        local_files_only=local_files_only,
        resume_download=resume_download,
        padding_side=padding_side,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id  # For finetuning
    return tokenizer

def get_model(
    config,
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    token=None,
    local_files_only=False,
    device=None,
    load_8bit=False,
    load_4bit=False,
    low_bit_mode=0,
    resume_download=True,
    trust_remote_code=True,
    offload_folder="model_offload_folder",
    revision=None,
    device_map="auto",
    cache_dir="~/.cache/huggingface/hub",
    lora_weights="",
):
    quantization_config = None
    if load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_8bit)

    # Additional configurations as necessary
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        use_auth_token=token,
        local_files_only=local_files_only,
        resume_download=resume_download,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    return model
