import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

# 1. Setup Command Line Arguments (for SLURM flexibility)
parser = argparse.ArgumentParser(description='GB10 Benchmark')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to run on (cpu or cuda)')
parser.add_argument('--iterations', type=int, default=200,
                    help='Number of training iterations to run')
args = parser.parse_args()

# 2. Configure Device
device = torch.device(args.device)
print(f"🚀 Starting training on: {device}")

# 3. Create a Simple Neural Network
# A larger hidden layer makes the GPU/CPU difference more obvious
model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 10)
).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Generate Synthetic Data (Coherent Memory in GB10 makes this fast!)
inputs = torch.randn(1024, 1024).to(device)
targets = torch.randn(1024, 10).to(device)

# 5. The Training Loop (Benchmarking)
iterations = int(args.iterations)
print(f"⏱️  Running {iterations} training iterations...")
start_time = time.time()

progress_interval = max(1, iterations // 5)

for i in range(iterations):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % progress_interval == 0:
        print(f"   Iteration {i+1}/{iterations} complete")

end_time = time.time()
total_time = end_time - start_time

print("-" * 30)
print(f"✅ Training Finished on {device.type.upper()}")
print(f"⏱️  Total Time: {total_time:.4f} seconds")
print("-" * 30)