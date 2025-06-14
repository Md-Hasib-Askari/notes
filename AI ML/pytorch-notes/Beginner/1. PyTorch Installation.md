### 🧱 Topic 1: PyTorch Installation (CPU/GPU)

**🧭 Overview**

PyTorch supports installation on various platforms with both CPU and GPU acceleration (CUDA). You can install it via pip, conda, or from source.


---

### ⚙️ 1. Check System Requirements

**✅ Python**

- Version: Python 3.8 to 3.12 supported.
- Recommended: Python ≥3.10 for best compatibility.


**✅ CUDA (for GPU support)**

- NVIDIA GPU with CUDA Compute Capability ≥3.5.
- CUDA toolkit installed (optional if using pip/conda-prebuilt binaries).


> 📌 Use NVIDIA CUDA GPUs list to check your GPU compatibility.



**✅ Drivers**

- Latest NVIDIA GPU driver installed.
- Linux/macOS/Windows all supported.



---

### 💾 2. Installation Options

**✅ Using pip (Recommended for most users)**
- CPU-only (lightweight)
- ```pip install torch torchvision torchaudio```
- GPU (CUDA 12.1)

```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Change cu121 to cu118 if using CUDA 11.8:

```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ Make sure your CUDA version matches the installation command.


**✅ Using conda (Anaconda/Miniconda users)**

**CPU-only**

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**GPU (CUDA 11.8)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


---

### 🧪 3. Verify Installation

Run the following Python code:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

✅ Expected output (GPU example):
```bash
PyTorch version: 2.x.x
CUDA available: True
CUDA device count: 1
Current CUDA device: NVIDIA GeForce RTX 3060
```

---

### 🛠️ 4. Optional: Installing Jupyter Notebook

```bash
pip install notebook
jupyter notebook
```

Or, if using Conda:

```bash
conda install notebook
```

---

### 🧰 5. Troubleshooting Tips

| Problem |	Solution |
| --- | --- |
| torch.cuda.is_available() returns False	| Ensure NVIDIA drivers and CUDA toolkit are properly installed. Use matching CUDA version. |
| Slow install	| Use a mirror or switch to conda. |
| Compatibility errors	| Create a fresh virtual environment. |

---

**🧪 Useful Links**

- PyTorch Install Guide (Official)
- CUDA Compatibility Table
- torch.cuda documentation
