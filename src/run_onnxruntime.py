import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(img.size[1] * 0.4), img.size[0], img.size[1]))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

ort_session = ort.InferenceSession("models/model.onnx")

dummy_img = Image.new("RGB", (455, 256))
dummy_img = transform(dummy_img).unsqueeze(0).numpy()

for _ in range(10):
    ort_session.run(None, {"image": dummy_img})

print("Benchmarking ONNX Runtime")
num_runs = 100
start = time.time()
for _ in range(num_runs):
    output = ort_session.run(None, {"image": dummy_img})
elapsed = time.time() - start

avg_latency = (elapsed / num_runs) * 1000
throughput = num_runs / elapsed

print(f"avg latency: {avg_latency:.2f}ms")
print(f"throughput: {throughput:.2f} images/sec")