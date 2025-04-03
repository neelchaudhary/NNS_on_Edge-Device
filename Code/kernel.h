#ifndef KERNEL_H
#define KERNEL_H

#include <cmath>
#include "heap.h"
// Kernel function declarations
__global__ void knn_bruteforce_kernel(const float *d_data, int n, int d, int k, int *d_neighbors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load current vector into registers
    // Assuming d=128 is manageable in registers (for large d consider another approach)
    float current[128];
    for (int i = 0; i < d; i++) {
        current[i] = d_data[idx * d + i];
    }

    // Max-heap for k neighbors
    float dist_heap[100]; // Ensure k <= 64 for simplicity
    int idx_heap[100];    
    int heap_size = 0; 

    for (int j = 0; j < n; j++) {
        if (j == idx) continue;
        float dist = 0.0f;
        for (int dim = 0; dim < d; dim++) {
            float diff = current[dim] - d_data[j * d + dim];
            dist += diff * diff;
        } 
        // Insert into heap
        heap_insert(dist_heap, idx_heap, k, dist, j, &heap_size);
    }

    // Now dist_heap and idx_heap contain the top-k neighbors (smallest distances)
    // dist_heap[0] is the largest distance among the chosen k
    // dist_heap is a max-heap of the k smallest distances

    // Copy to global memory (unsorted)
    // Optionally, you could sort them here, but it's not required if you just need them.
    for (int i = 0; i < k; i++) d_neighbors[idx * k + i] = idx_heap[i];
}
#endif // KERNEL_H
