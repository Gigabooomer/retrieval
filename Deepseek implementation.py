import warnings
from typing import Optional
from threading import Thread
import torch
import asyncio
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    set_seed,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from utils import H2OTextIteratorStreamer, clear_torch_cache

set_seed(42)
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

def get_config(base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    config = AutoConfig.from_pretrained(base_model)
    return config

def get_tokenizer(
    pretrained_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_files_only=False,
    resume_download=True,
    token=None,
    padding_side="left",
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        local_files_only=local_files_only,
        resume_download=resume_download,
        token=token,
        padding_side=padding_side,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id  # For finetuning
    return tokenizer

def get_streamer(tokenizer, iterator=True, skip_prompt=True, h2o=False):
    if h2o:
        streamer = H2OTextIteratorStreamer(
            tokenizer, skip_prompt=skip_prompt, block=False
        )
        return streamer
    if iterator:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=skip_prompt
        )
    else:
        streamer = TextStreamer(tokenizer, pt=skip_prompt)
    return streamer

def get_model(
    config,
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_files_only=False,
    device=None,
    load_8bit=False,
    load_4bit=False,
    low_bit_mode=0,
    resume_download=True,
    token=None,
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
    elif low_bit_mode > 0:
        quantization_config = BitsAndBytesConfig(load_in_4bit=load_4bit, load_in_8bit=load_8bit)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if "llama" in base_model:
        torch_dtype = torch.bfloat16

    model_kwargs = {
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "resume_download": resume_download,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "offload_folder": offload_folder,
        "revision": revision,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }
    model = AutoModelForCausalLM.from_pretrained(
        base_model, config=config, **model_kwargs
    )
    return model

def get_pipeline(
    model,
    tokenizer,
    streamer,
    torch_dtype=torch.float16,
    task="text-generation",
    min_new_tokens=0,
    max_new_tokens=128,
    return_full_text=False,
    temperature=0.2,
    do_sample=False,
):
    eos_token_id = tokenizer.eos_token_id
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        streamer=streamer,
        task=task,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        return_full_text=return_full_text,
        do_sample=do_sample,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    return pipe

def get_langchain_compatible_llm(pipe):
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
