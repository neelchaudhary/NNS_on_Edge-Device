// Device function declarations
#ifndef HEAP_H
#define HEAP_H

#include <cuda_runtime.h>
#include <cmath>

__device__ __inline__ void heapify_down(float* dist_heap, int* idx_heap, int heap_size, int root) {
    int largest = root;
    int left = 2 * root + 1;
    int right = 2 * root + 2;

    if (left < heap_size && dist_heap[left] > dist_heap[largest]) largest = left;
    if (right < heap_size && dist_heap[right] > dist_heap[largest]) largest = right;

    if (largest != root) {
        float tmp_dist = dist_heap[root];
        dist_heap[root] = dist_heap[largest];
        dist_heap[largest] = tmp_dist;

        int tmp_idx = idx_heap[root];
        idx_heap[root] = idx_heap[largest];
        idx_heap[largest] = tmp_idx;

        heapify_down(dist_heap, idx_heap, heap_size, largest);
    }
}

__device__ __inline__ void heapify_up(float* dist_heap, int* idx_heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (dist_heap[i] > dist_heap[parent]) {
            float tmp_dist = dist_heap[i];
            dist_heap[i] = dist_heap[parent];
            dist_heap[parent] = tmp_dist;

            int tmp_idx = idx_heap[i];
            idx_heap[i] = idx_heap[parent];
            idx_heap[parent] = tmp_idx;

            i = parent;
        } else {
            break;
        }
    }
}

__device__ __inline__ void heap_insert(float* dist_heap, int* idx_heap, int k, float dist, int idx, int* heap_size) {
    if (*heap_size < k) {
        dist_heap[*heap_size] = dist;
        idx_heap[*heap_size] = idx;
        heapify_up(dist_heap, idx_heap, *heap_size);
        (*heap_size)++;
    } else {
        if (dist < dist_heap[0]) {
            dist_heap[0] = dist;
            idx_heap[0] = idx;
            heapify_down(dist_heap, idx_heap, k, 0);
        }
    }
}

__device__ __inline__ void heap_delete(float *dist_heap, int *idx_heap, int &heap_size, float &popped_dist, int &popped_idx) {
    if (heap_size == 0) return; // Heap is empty

    // Remove the root (max element)
    popped_dist = dist_heap[0];
    popped_idx = idx_heap[0];

    // Replace root with last element
    dist_heap[0] = dist_heap[heap_size - 1];
    idx_heap[0] = idx_heap[heap_size - 1];

    // Decrease heap size
    heap_size--;

    // Restore heap property
    heapify_down(dist_heap, idx_heap, heap_size, 0);
}


#endif