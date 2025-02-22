# benchmark.py
import time
import pandas as pd
import threading
from test_cases import get_test_cases
from gpu_monitor import monitor_gpu, get_gpu_metrics, monitoring
from models import get_model
import ace_tools as tools

def benchmark_model(model_name: str, test_category="nl_to_sql"):
    """Runs benchmark on a locally deployed LLM with real-time GPU monitoring."""
    llm = get_model(model_name)
    test_cases = get_test_cases(test_category)

    results = []

    for case in test_cases:
        query = case["query"]
        expected = case["expected"]

        # Start GPU monitoring in a separate thread
        global monitoring
        monitoring = True
        gpu_thread = threading.Thread(target=monitor_gpu)
        gpu_thread.start()

        # Measure inference time
        start_time = time.time()
        response = llm.predict(query)
        latency = time.time() - start_time

        # Stop GPU monitoring
        monitoring = False
        gpu_thread.join()

        # Get final GPU stats
        final_gpu_util, final_memory, final_power = get_gpu_metrics()

        # Compute accuracy (simple word match)
        accuracy = 1 if expected.lower() in response.lower() else 0

        # Store benchmark results
        results.append({
            "Model": model_name,
            "Query": query,
            "Response": response,
            "Expected": expected,
            "Latency (s)": round(latency, 3),
            "Accuracy": accuracy,
            "Final GPU Utilization (%)": final_gpu_util,
            "Final Memory Used (MB)": final_memory,
            "Final Power Consumption (W)": final_power,
        })

    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    tools.display_dataframe_to_user(name="Local Model Benchmark Results", dataframe=df)

# Run benchmark
if __name__ == "__main__":
    benchmark_model("mistral-7b", "nl_to_sql")
