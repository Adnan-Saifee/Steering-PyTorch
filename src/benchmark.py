import os
import time
import torch
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core
from PIL import Image
from torchvision import transforms
import csv

from model import SteeringNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(img.size[1] * 0.4), img.size[0], img.size[1]))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

dummy_img = Image.new("RGB", (455, 256))
dummy_input = transform(dummy_img).unsqueeze(0)
dummy_numpy = dummy_input.numpy()

results = {}

print("=" * 60)
print("benchmarking PyTorch model")
print("=" * 60)
pytorch_model = SteeringNet().to(device)
pytorch_model.load_state_dict(torch.load("models/checkpoints/best.pt", map_location=device))
pytorch_model.eval()

with torch.no_grad():
    for _ in range(10):
        pytorch_model(dummy_input.to(device))

num_runs = 100
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        pytorch_model(dummy_input.to(device))
elapsed = time.time() - start

pytorch_latency = (elapsed / num_runs) * 1000
pytorch_size = os.path.getsize("models/checkpoints/best.pt") / (1024 * 1024)

results["PyTorch"] = {
    "latency_ms": pytorch_latency,
    "size_mb": pytorch_size,
}
print(f"latency: {pytorch_latency:.2f}ms")
print(f"model size: {pytorch_size:.2f}MB")

print("\n" + "=" * 60)
print("benchmarking ONNX Runtime")
print("=" * 60)
ort_session = ort.InferenceSession("models/model.onnx")

for _ in range(10):
    ort_session.run(None, {"image": dummy_numpy})

start = time.time()
for _ in range(num_runs):
    ort_session.run(None, {"image": dummy_numpy})
elapsed = time.time() - start

onnx_latency = (elapsed / num_runs) * 1000
onnx_size = os.path.getsize("models/model.onnx") / (1024 * 1024)

results["ONNX Runtime"] = {
    "latency_ms": onnx_latency,
    "size_mb": onnx_size,
}
print(f"latency: {onnx_latency:.2f}ms")
print(f"model size: {onnx_size:.2f}MB")

print("\n" + "=" * 60)
print("benchmarking ONNX Runtime (Quantized)")
print("=" * 60)
quantized_session = ort.InferenceSession("models/model_quantized.onnx")

for _ in range(10):
    quantized_session.run(None, {"image": dummy_numpy})

start = time.time()
for _ in range(num_runs):
    quantized_session.run(None, {"image": dummy_numpy})
elapsed = time.time() - start

quantized_latency = (elapsed / num_runs) * 1000
quantized_size = os.path.getsize("models/model_quantized.onnx") / (1024 * 1024)

results["ONNX Quantized"] = {
    "latency_ms": quantized_latency,
    "size_mb": quantized_size,
}
print(f"latency: {quantized_latency:.2f}ms")
print(f"model size: {quantized_size:.2f}MB")

print("\n" + "=" * 60)
print("benchmarking OpenVINO")
print("=" * 60)
ie = Core()
compiled_model = ie.compile_model("models/model_openvino.xml", "CPU")
infer_request = compiled_model.create_infer_request()

for _ in range(10):
    infer_request.infer({"image": dummy_numpy})

start = time.time()
for _ in range(num_runs):
    infer_request.infer({"image": dummy_numpy})
elapsed = time.time() - start

openvino_latency = (elapsed / num_runs) * 1000
openvino_size = (os.path.getsize("models/model_openvino.xml") + os.path.getsize("models/model_openvino.bin")) / (1024 * 1024)

results["OpenVINO"] = {
    "latency_ms": openvino_latency,
    "size_mb": openvino_size,
}
print(f"latency: {openvino_latency:.2f}ms")
print(f"model size: {openvino_size:.2f}MB")

print("\n" + "=" * 60)
print("summary")
print("=" * 60)

with open("results/benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "latency_ms", "size_mb", "speedup_vs_pytorch", "size_reduction"])
    for name, metrics in results.items():
        speedup = pytorch_latency / metrics["latency_ms"]
        size_reduction = (1 - metrics["size_mb"] / pytorch_size) * 100
        writer.writerow([name, f"{metrics['latency_ms']:.2f}", f"{metrics['size_mb']:.2f}", f"{speedup:.2f}x", f"{size_reduction:.1f}%"])
        print(f"{name:20s} | {metrics['latency_ms']:7.2f}ms | {metrics['size_mb']:6.2f}MB | {speedup:5.2f}x faster | {size_reduction:5.1f}% smaller")

print(f"\nresults saved to results/benchmark_results.csv")