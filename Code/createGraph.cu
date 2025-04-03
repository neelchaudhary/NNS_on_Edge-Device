#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "kernel.h" 

using namespace std;
#define CHECK_CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
      exit(code);
   }
}

void save_ivecs(const std::string& filename, const std::vector<std::vector<int>>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (const auto& vec : data) {
        int dim = vec.size();
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int));  // Write dimension
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(int));  // Write vector data
    }

    file.close();
    std::cout << "Saved neighbors to " << filename << " successfully.\n";
}

std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<float>> data;
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (true) {
        int dim;
        // Read dimension (4 bytes)
        file.read(reinterpret_cast<char*>(&dim), 4);
        if (!file.good()) break;
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file.good()) break;
        data.push_back(vec);
    }
    file.close();
    return data;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& data) {
    std::vector<float> flat_data;
    for (auto &v : data) {
        flat_data.insert(flat_data.end(), v.begin(), v.end());
    }
    return flat_data;
}

int main(){
    std::string baseFile = "sift/sift_base.fvecs";
    int neighbors = 100;           // number of neighbors
    int n = 1000000; 
    int d = 128; 
    std::vector<std::vector<float>> data = read_fvecs(baseFile);
    if ((int)data.size() < n) {
        std::cerr << "File has fewer than n vectors. Using " << data.size() << " instead of " << n << ".\n";
        n = (int)data.size();
    }
    std::vector<float> flat_data = flatten(data);
    float *d_data;
    CHECK_CUDA_ERR(cudaMalloc(&d_data, n * d * sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_data, flat_data.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));
    printf("Load the data!\n");
    int *d_neighbors;
    CHECK_CUDA_ERR(cudaMalloc(&d_neighbors, n * neighbors * sizeof(int)));
    // Launch kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    knn_bruteforce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, d, neighbors, d_neighbors);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

    std::vector<int> h_neighbors(n * neighbors);
    cudaMemcpy(h_neighbors.data(), d_neighbors, n * neighbors * sizeof(int), cudaMemcpyDeviceToHost);
    // Organize into 2D vector
    std::vector<std::vector<int>> neighbors_2d(n, std::vector<int>(neighbors));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < neighbors; j++) {
            neighbors_2d[i][j] = h_neighbors[i * neighbors + j];
        }
    }

    // Save to ivecs file
    save_ivecs("sift/neighbors.ivecs", neighbors_2d);
    printf("Save the neighbors!\n");

    // Free GPU memory
    cudaFree(d_neighbors);

    return 0;

}