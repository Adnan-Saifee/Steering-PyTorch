# Steering Angle Prediction

CNN for steering angle prediction from road images with ONNX and OpenVINO optimization.

## Setup

GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage

Train:
```bash
python src/train.py
```

Evaluate:
```bash
python src/evaluate.py
```

Export and benchmark:
```bash
python src/export_onnx.py
python src/quantize.py
python src/benchmark.py
```

## Results

Test MAE: 11.84 degrees

Inference benchmarks (GTX 1660 Ti, 100 runs):
- PyTorch: 0.84ms
- ONNX Runtime: 0.28ms (2.98x faster)
- OpenVINO: 0.30ms (2.78x faster, 46.9% smaller)
- Quantized: 3.98ms (73.7% smaller)

## Data

60k+ dashcam images from Sully Chen's dataset. Time-sequential 70/15/15 train/val/test split. Cropped top 40% (sky), resized to 66x200.

## Model

5-layer CNN, 50k parameters, 0.97MB. Trained for 20 epochs with validation monitoring.

High MAE due to class imbalance (mostly straight driving). Future: data augmentation, angle-weighted loss.
