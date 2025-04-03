# 🔍 Nearest Neighbor Search on GPU (CUDA + C++)

A high-performance **Approximate Nearest Neighbor Search (ANN)** implementation optimized for **edge devices** using **CUDA** and **C++**. The project is designed to accelerate high-dimensional similarity search while minimizing memory usage and maximizing throughput on GPU hardware.

---

## 🚀 Features

- ✅ **CUDA-accelerated k-NN search**
- ✅ Memory-efficient **min-max heap** with bounded priority queue
- ✅ **1-bit random projections** for dimensionality reduction
- ✅ Optimized **shared memory & coalesced memory access**
- ✅ Support for **batch queries** and **real-time search**
- ✅ Modular architecture for easy integration into ML pipelines

---

## 📁 Project Structure


---

## 📦 Dependencies

- CUDA >= 11.0
- C++17
- CMake >= 3.10
- (Optional) Python bindings using `pybind11` for hybrid usage

---

## 🛠️ Build Instructions

```bash
# Clone the repo
git clone https://github.com/neelchaudhary/nns_gpu_project.git
cd nns_gpu_project

# Create build directory
mkdir build && cd build

# Run CMake
cmake ..

# Build the project
make -j$(nproc)
