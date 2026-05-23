# Context2Context: A Zero-Shot SAR Image Speckle Filter

PyTorch reproduction of the paper:

> Context2Context: A Zero-Shot SAR Image Speckle Filter

This project reproduces the zero-shot SAR image despeckling method proposed in the paper **Context2Context: A Zero-Shot SAR Image Speckle Filter**.  
The method suppresses speckle noise in SAR images without requiring paired clean-noisy datasets or large-scale supervised training.

---

# Project Structure

```bash
.
├── data/
│   ├── sample_slc_image.npy                # Complex SAR image (.npy)
│   └── mat2npy.py                # Convert .mat SAR data to .npy
│
├── models/
│
├── results/
│   └── *.npy                     # Inference results (.npy)
│
├── utils/
│   ├── dataset.py                # Dataset loading script
│   └── ccm_search.py                    # Context-to-Context module
│
├── visualize_results/
│   └── *.png                     # Visualized despeckling results
│
├── train.py                      # Training script
├── infer.py                      # Inference script
├── visualize.py                  # Visualization script
│
└── README.md
```

---

# Features

- Zero-shot SAR image despeckling
- No clean reference image required
- Supports complex-valued SAR image input
- NumPy-based SAR data pipeline
- Self-supervised optimization
- Result visualization support
- Modular PyTorch implementation

---

# Environment

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- SciPy
- OpenCV
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib opencv-python
```

---

# Dataset Preparation

The `data/` directory stores SAR images in `.npy` format.

## Input Format

Input SAR image should be:

- Complex-valued SAR image
- Stored as NumPy array (`.npy`)
- Example shape:

```python
(H, W)      # complex64 or complex128
```

Example:

```python
import numpy as np

img = np.load("data/sample.npy")

print(img.shape)
print(img.dtype)
```

---

# Convert `.mat` to `.npy`

If the SAR image is stored in MATLAB `.mat` format, use:

```bash
python data/mat2npy.py
```

This script converts `.mat` SAR data into `.npy` format for training and inference.

---

# Training

Run the training script:

```bash
python train.py
```

The training pipeline will:

- Load SAR image data
- Construct the C2C network
- Perform self-supervised optimization
- Generate despeckled outputs

---

# Inference

Run inference:

```bash
python infer.py
```

Inference outputs will be saved to:

```bash
results/
```

Output format:

```bash
*.npy
```

---

# Visualization

To visualize `.npy` despeckling results:

```bash
python visualize.py
```

Visualization images will be saved to:

```bash
visualize_results/
```

Visualization examples may include:

- Original SAR image
- Despeckled SAR image
- Intensity image
- Log-intensity rendering

---

# Model Architecture

Network implementations are located in:

```bash
models/
```

Main components include:

- Feature extraction blocks
- Context aggregation modules
- Context-to-Context mapping
- Self-supervised reconstruction framework

---

# Utilities

Utility scripts are located in:

```bash
utils/
```

## dataset.py

Responsible for:

- Loading SAR `.npy` data
- Patch extraction
- Data preprocessing


# Results

Inference outputs are stored in:

```bash
results/
```

Visualization results are stored in:

```bash
visualize_results/
```

Typical results demonstrate:

- Effective speckle suppression
- Edge and structure preservation
- Improved SAR image quality

---

# Example Workflow

## 1. Prepare Data

Place SAR `.npy` image into:

```bash
data/
```

---

## 2. Train Model

```bash
python train.py
```

---

## 3. Run Inference

```bash
python infer.py
```

---

## 4. Visualize Results

```bash
python visualize.py
```

---

# Acknowledgements

This project is based on the paper:

> Context2Context: A Zero-Shot SAR Image Speckle Filter

Implemented using:

- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

# Notes

- Input SAR images are assumed to be complex-valued.
- The project supports single-image zero-shot optimization.
- Visualization is performed on intensity or log-intensity images.
- Results may vary depending on SAR sensor type and image quality.