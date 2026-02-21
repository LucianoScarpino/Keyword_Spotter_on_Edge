# Keyword Spotter (KWS) on Edge — MFCC (ONNX) + Tiny CNN (ONNX/INT8)

Compact keyword-spotting pipeline designed for edge deployment. Audio is standardized (mono, **16 kHz**, **1 s** window), converted to **MFCC** by an **ONNX frontend**, then classified by a lightweight **CNN** exported as a separate **ONNX** graph. The project includes training, ONNX export, and static **INT8** quantization (QDQ + entropy calibration + per-channel weights).  [oai_citation:0‡report.pdf](sediment://file_000000001d487246a20043dbc8a97eff)

## Why this project (portfolio angle)
- End-to-end edge-oriented ML pipeline (feature extraction → model → export → quantization)
- Concrete constraints: model size + latency + confidence thresholding (pass@0.999)
- Clear metrics and reproducible setup

---

## Pipeline
1. **Waveform preprocessing**: fixed-length 1 s, 16 kHz
2. **Augmentation (train only)**: time shift + gain jitter + mild noise (SNR-controlled)
3. **Feature extraction**: MFCC
4. **Classifier**: compact CNN with depthwise-separable residual blocks
5. **Export**: two ONNX graphs (frontend + model)
6. **Quantization**: static INT8 (QDQ)

---

## Final configuration (selected)
### MFCC
- Sampling rate: **16 kHz**
- Frame length: **0.032 s** (512 samples)
- Frame step: **0.032 s** (no overlap)
- Frames per 1 s: **31**
- n_mels: **20**
- n_mfcc: **20**
- fmin / fmax: **40 Hz / 6000 Hz**  [oai_citation:1‡report.pdf](sediment://file_000000001d487246a20043dbc8a97eff)

### Training
- Optimizer: Adam, lr **1e-3**, wd **0**
- Epochs: **30**, best checkpoint by validation accuracy
- Batch size: **64**
- Scheduler: CosineAnnealing (ηmin **1e-5**)
- BN freeze: epoch **6**
- Seeds: torch/numpy/python = **0**  [oai_citation:2‡report.pdf](sediment://file_000000001d487246a20043dbc8a97eff)

### Model
- Stem conv **3×3** (stride (2,1) to downsample frequency only)
- **5** depthwise-separable residual blocks
- Global average pooling + linear classifier
- Width multiplier **w = 0.28** (controls channel scaling)  [oai_citation:3‡report.pdf](sediment://file_000000001d487246a20043dbc8a97eff)

---

## Results (edge-focused)
- Test accuracy: **99.50%**
- ONNX Float32 size: **198.2 KB** (frontend 40.5 KB + model 157.7 KB)
- ONNX INT8 size: **124.6 KB** (model 84.1 KB total)
- End-to-end latency (approx.): **5.0 ms** (3.4 ms model + 1.5 ms frontend)
- pass@0.999 (correct): **0.28** (PyTorch), **0.28** (ONNX Float32), **0.275** (ONNX INT8)  [oai_citation:4‡report.pdf](sediment://file_000000001d487246a20043dbc8a97eff)

> pass@0.999 measures the fraction of samples correctly classified with max softmax probability > 0.999 (relevant when using a high-confidence trigger).

---

## Project structure (key files)
- `main.py`: training + evaluation + ONNX export + calls quantization
- `custom_dataset.py`: dataset loader
- `kernel.py`: MFCC builder + augmentation utilities
- `training.py`: training loop (checkpoint selection by val accuracy)
- `optimization.py`: ONNX quantization (static INT8)

Exports are saved in:
- `saved_models/<timestamp>_frontend.onnx`
- `saved_models/<timestamp>_model.onnx`
- `saved_models/results.csv` (appends run metadata)

---

## How to run
### 1) Train + export + quantize
```bash
python main.py \
  --sampling-rate 16000 \
  --frame-length 0.032 --frame-step 0.032 \
  --n-mels 20 --n-mfcc 20 \
  --f-min 40 --f-max 6000 \
  --lr 1e-3 --weight-decay 0 \
  --model-width-mult 0.28 \
  --seed 0 --train-batch-size 64


