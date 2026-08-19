import os
import time
import numpy as np
from openvino.runtime import Core
from PIL import Image
from torchvision import transforms

ir_model_path = "models/model_openvino.xml"

if not os.path.exists(ir_model_path):
    print("converting ONNX to OpenVINO IR format...")
    from openvino.tools import mo
    mo.convert_model("models/model.onnx", output_dir="models/")
    print("conversion complete")

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(img.size[1] * 0.4), img.size[0], img.size[1]))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

ie = Core()
compiled_model = ie.compile_model(ir_model_path, "CPU")
infer_request = compiled_model.create_infer_request()

dummy_img = Image.new("RGB", (455, 256))
dummy_img = transform(dummy_img).unsqueeze(0).numpy()

for _ in range(10):
    infer_request.infer({"image": dummy_img})

print("Benchmarking OpenVINO")
num_runs = 100
start = time.time()
for _ in range(num_runs):
    infer_request.infer({"image": dummy_img})
    output = infer_request.get_output_tensor().data
elapsed = time.time() - start

avg_latency = (elapsed / num_runs) * 1000
throughput = num_runs / elapsed

print(f"avg latency: {avg_latency:.2f}ms")
print(f"throughput: {throughput:.2f} images/sec")