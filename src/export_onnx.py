import torch
import onnx
import onnxruntime as ort
import numpy as np

from model import SteeringNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SteeringNet().to(device)
model.load_state_dict(torch.load("models/checkpoints/best.pt", map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, 66, 200).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=["image"],
    output_names=["steering_angle"],
    opset_version=12,
    do_constant_folding=True,
)

print("exported to models/model.onnx")

onnx_model = onnx.load("models/model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

with torch.no_grad():
    pytorch_output = model(dummy_input).cpu().numpy()

ort_session = ort.InferenceSession("models/model.onnx")
onnx_output = ort_session.run(None, {"image": dummy_input.cpu().numpy()})
onnx_output = np.array(onnx_output[0])

max_diff = np.max(np.abs(pytorch_output - onnx_output))
print(f"max difference between PyTorch and ONNX: {max_diff:.6f}")

if max_diff < 1e-4:
    print("parity check passed")
else:
    print("warning: outputs differ significantly")