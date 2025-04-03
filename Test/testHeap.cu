#include <iostream>
#include <cuda_runtime.h>
#include "heap.h"

#define CHECK_CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

__global__ void testHeapKernel() {
    const int k = 5; // Max size of the heap
    float dist_heap[k];
    int idx_heap[k];
    int heap_size = 0;

    // Test data
    float distances[] = {10.0f, 20.0f, 5.0f, 8.0f, 15.0f, 3.0f, 25.0f};
    int indices[] = {0, 1, 2, 3, 4, 5, 6};
    int num_elements = sizeof(distances) / sizeof(float);

    // Insert elements into the heap
    for (int i = 0; i < num_elements; i++) {
        heap_insert(dist_heap, idx_heap, k, distances[i], indices[i], &heap_size);
    }

    // Output the heap content after insertion
    printf("Heap after insertion:\n");
    for (int i = 0; i < heap_size; i++) {
        printf("dist: %.2f, idx: %d\n", dist_heap[i], idx_heap[i]);
    }

    // Perform deletions to test heap_delete
    // printf("\nPerforming deletions:\n");
    // while (heap_size > 0) {
    //     float popped_dist;
    //     int popped_idx;
    //     heap_delete(dist_heap, idx_heap, heap_size, popped_dist, popped_idx);
    //     printf("Popped dist: %.2f, idx: %d\n", popped_dist, popped_idx);
    // }
}


int main() {
    // Launch the test kernel
    testHeapKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Check for CUDA errors
    CHECK_CUDA_ERR(cudaGetLastError());

    return 0;
}
