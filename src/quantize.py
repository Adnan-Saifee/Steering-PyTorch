import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

# Applying int8 quantization to shrink model by converint weights from 32-float to 8-bit integers
# Makes model smaller and maybe faster? without losing much accuracy
print("Applying dynamic INT8 quantization")
quantize_dynamic(
    "models/model.onnx",
    "models/model_quantized.onnx",
    weight_type=QuantType.QInt8,
)
print("quantized model saved to models/model_quantized.onnx")

dummy_input = np.random.randn(1, 3, 66, 200).astype(np.float32)

print("\nverifying accuracy after quantization...")
original_session = ort.InferenceSession("models/model.onnx")
quantized_session = ort.InferenceSession("models/model_quantized.onnx")

original_output = original_session.run(None, {"image": dummy_input})
quantized_output = quantized_session.run(None, {"image": dummy_input})

original_output = np.array(original_output[0])
quantized_output = np.array(quantized_output[0])

max_diff = np.max(np.abs(original_output - quantized_output))
print(f"max difference after quantization: {max_diff:.6f}")

if max_diff < 0.1:
    print("quantization successful - minimal accuracy loss")
else:
    print("warning: quantization caused larger accuracy shift")