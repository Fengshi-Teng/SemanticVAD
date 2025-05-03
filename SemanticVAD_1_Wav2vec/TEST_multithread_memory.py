import torch
import torchaudio
import concurrent.futures
import psutil
import time
import matplotlib.pyplot as plt
import random
import gc
import os
import pandas as pd

model_path = "./models/SemanticVAD(2*512*256_4000)1745813110.pt" 
test_data = "./dataset/binary_classification/test/test.csv"
df = pd.read_csv(test_data).sample(n=50, random_state=int(time.time()))

# ====== Load Model and Feature Extractor ======
from transformers import AutoFeatureExtractor, AutoModel
from train import DistilHuBERTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = "ntu-spml/distilhubert"
extractor = AutoFeatureExtractor.from_pretrained(base_model)

encoder = AutoModel.from_pretrained(base_model)
NUM_LABELS = 2
model = DistilHuBERTClassifier(base_model=encoder, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

process = psutil.Process(os.getpid())

# ====== Prepare dataset ======
file_list = list(df['filepath'])  # Assume your CSV is already loaded
assert len(file_list) > 0, "No audio files found!"

# ====== Define Single Inference Function ======
def single_inference(file_path):
    """
    Perform inference on a single audio file.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"]

    with torch.no_grad():
        _ = model(input_values=input_values)

# ====== Define Thread Worker Function ======
def thread_worker(thread_id, memory_log, file_list):
    """
    Each thread runs 50 rounds of inference independently and records memory usage after each inference.
    """
    local_memory_log = []
    for round_idx in range(50):  # Each thread performs 50 inferences
        file_path = random.choice(file_list)

        before = process.memory_info().rss / 1024 / 1024  # Memory in MB before inference

        single_inference(file_path)

        after = process.memory_info().rss / 1024 / 1024  # Memory in MB after inference
        used = after - baseline_memory
        local_memory_log.append(used)

        gc.collect()  # Garbage collection to minimize interference

    memory_log[thread_id] = local_memory_log

# ====== Main Experiment Setup ======
thread_counts = [2, 4, 6, 8, 10, 12]
rounds_per_thread = 50

# Measure baseline memory after model loading (before any inference)
baseline_memory = process.memory_info().rss / 1024 / 1024  # In MB
print(f"Baseline memory after loading model: {baseline_memory:.2f} MB")

# Store memory usage results for each thread configuration
all_results = {}

for num_threads in thread_counts:
    print(f"\n🚀 Starting experiment with {num_threads} threads...")

    memory_log = {}

    # Launch multiple threads (each running 50 rounds independently)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(thread_worker, tid, memory_log, file_list) for tid in range(num_threads)]
        concurrent.futures.wait(futures)

    # Average memory usage per round across all threads
    avg_memory_per_round = [0] * rounds_per_thread
    for round_idx in range(rounds_per_thread):
        total = 0
        for tid in memory_log:
            total += memory_log[tid][round_idx]
        avg_memory_per_round[round_idx] = total / num_threads

    all_results[num_threads] = avg_memory_per_round

    print(f"✅ Finished experiment with {num_threads} threads.")

# ====== Plotting the Memory Usage Curves ======
plt.figure(figsize=(12, 6))
for num_threads, memory_curve in all_results.items():
    plt.plot(range(1, rounds_per_thread + 1), memory_curve, label=f'{num_threads} Threads')

plt.title("Memory Usage Growth during Parallel Inference")
plt.xlabel("Inference Round")
plt.ylabel("Memory Increase over Baseline (MB)")
plt.legend()
plt.grid(True)
plt.show()