import requests
import time

# Set the API URL for the deployed DeepSeek model
API_URL = "http://localhost:8500/generate"  # Change if running on a remote server

# Define test prompts
test_prompts = [
    "Tell me a joke about AI.",
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the future of technology.",
    "Summarize the benefits of AI in healthcare."
]

# Define max tokens for each test
MAX_TOKENS = 100

# Test Results Storage
results = []

# Start testing
print("Running inference test on DeepSeek model...\n")

for prompt in test_prompts:
    payload = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS
    }

    try:
        print(f"Testing prompt: \"{prompt[:30]}...\"")

        start_time = time.time()
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        end_time = time.time()

        latency = round(end_time - start_time, 3)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            generated_tokens = len(generated_text.split())  # Approximate token count
            tokens_per_sec = round(generated_tokens / latency, 2) if latency > 0 else 0

            # Store results in a list
            results.append((prompt, latency, generated_tokens, tokens_per_sec))

            print(f"Response received in {latency}s | Tokens: {generated_tokens} | Speed: {tokens_per_sec} tokens/sec")
            print(f"Model Response:\n{generated_text}\n")
            print("-" * 80)

        else:
            print(f"Test Failed! Error {response.status_code}: {response.text}\n")

    except requests.exceptions.RequestException as e:
        print(f"Model Test Failed! Could not reach the API.\nError: {str(e)}\n")

# Print test summary
print("\nTest Summary:")
print("-" * 80)
print(f"{'Prompt':<50} {'Latency (s)':<12} {'Tokens':<10} {'Tokens/sec'}")
print("-" * 80)
for prompt, latency, tokens, tokens_per_sec in results:
    print(f"{prompt[:47]+'...' if len(prompt) > 47 else prompt:<50} {latency:<12} {tokens:<10} {tokens_per_sec}")
print("-" * 80)
