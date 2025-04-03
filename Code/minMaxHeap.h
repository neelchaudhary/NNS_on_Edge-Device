#ifndef MINMAXHEAP_H
#define MINMAXHEAP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Macros as defined
#define PARENT(idx) ((idx - 1) / 2)
#define LEFT_CHILD(idx) (2 * idx + 1)
#define RIGHT_CHILD(idx) (2 * idx + 2)
#define GRANDPARENT(idx) ((idx - 3) / 4)

__device__ __inline__ bool isOnMinLevel(int idx) {
    int level = __float2int_rz(log2f(idx + 1));
    return level % 2 == 0;
}

__device__ __inline__ void swapPair(float* dist, int* idx, int i, int j) {
    float tempDist = dist[i];
    dist[i] = dist[j];
    dist[j] = tempDist;

    int tempIdx = idx[i];
    idx[i] = idx[j];
    idx[j] = tempIdx;
}

// Similar logic as the original code, but now using dist/idx arrays:
__device__ __inline__ void pushUpMin(float* dist, int* idx, int curIdx) {
    while (curIdx > 2 && GRANDPARENT(curIdx) >= 0 && dist[curIdx] < dist[GRANDPARENT(curIdx)]) {
        swapPair(dist, idx, curIdx, GRANDPARENT(curIdx));
        curIdx = GRANDPARENT(curIdx);
    }
}

__device__ __inline__ void pushUpMax(float* dist, int* idx, int curIdx) {
    while (curIdx > 2 && GRANDPARENT(curIdx) >= 0 && dist[curIdx] > dist[GRANDPARENT(curIdx)]) {
        swapPair(dist, idx, curIdx, GRANDPARENT(curIdx));
        curIdx = GRANDPARENT(curIdx);
    }
}

__device__ __inline__ void pushUpKernel(float* dist, int* idx, int curIdx) {
    if (curIdx == 0) return;
    int parentIdx = PARENT(curIdx);

    if (isOnMinLevel(curIdx)) {
        if (dist[curIdx] > dist[parentIdx]) {
            swapPair(dist, idx, curIdx, parentIdx);
            pushUpMax(dist, idx, parentIdx);
        } else {
            pushUpMin(dist, idx, curIdx);
        }
    } else {
        if (dist[curIdx] < dist[parentIdx]) {
            swapPair(dist, idx, curIdx, parentIdx);
            pushUpMin(dist, idx, parentIdx);
        } else {
            pushUpMax(dist, idx, curIdx);
        }
    }
}

__device__ __inline__ int smallestChildOrGrandchild(float* dist, int m, int size) {
    int bestIdx = -1;
    float smallestValue = 3.4e38f;

    int children[2] = {LEFT_CHILD(m), RIGHT_CHILD(m)};
    for (int i = 0; i < 2; i++) {
        int c = children[i];
        if (c < size && dist[c] < smallestValue) {
            smallestValue = dist[c];
            bestIdx = c;
        }
    }

    int grandchildren[4] = {
        LEFT_CHILD(LEFT_CHILD(m)), RIGHT_CHILD(LEFT_CHILD(m)),
        LEFT_CHILD(RIGHT_CHILD(m)), RIGHT_CHILD(RIGHT_CHILD(m))
    };
    for (int i = 0; i < 4; i++) {
        int gc = grandchildren[i];
        if (gc < size && dist[gc] < smallestValue) {
            smallestValue = dist[gc];
            bestIdx = gc;
        }
    }

    return bestIdx;
}

__device__ __inline__ int largestChildOrGrandchild(float* dist, int m, int size) {
    int bestIdx = -1;
    float largestValue = -3.4e38f;

    int children[2] = {LEFT_CHILD(m), RIGHT_CHILD(m)};
    for (int i = 0; i < 2; i++) {
        int c = children[i];
        if (c < size && dist[c] > largestValue) {
            largestValue = dist[c];
            bestIdx = c;
        }
    }

    int grandchildren[4] = {
        LEFT_CHILD(LEFT_CHILD(m)), RIGHT_CHILD(LEFT_CHILD(m)),
        LEFT_CHILD(RIGHT_CHILD(m)), RIGHT_CHILD(RIGHT_CHILD(m))
    };
    for (int i = 0; i < 4; i++) {
        int gc = grandchildren[i];
        if (gc < size && dist[gc] > largestValue) {
            largestValue = dist[gc];
            bestIdx = gc;
        }
    }

    return bestIdx;
}

__device__ __inline__ bool isGrandchild(int idx, int swapIdx) {
    int leftChild = 2 * idx + 1;
    int rightChild = 2 * idx + 2;
    int leftGrandchild1 = 2 * leftChild + 1;
    int leftGrandchild2 = 2 * leftChild + 2;
    int rightGrandchild1 = 2 * rightChild + 1;
    int rightGrandchild2 = 2 * rightChild + 2;

    return (swapIdx == leftGrandchild1 || swapIdx == leftGrandchild2 ||
            swapIdx == rightGrandchild1 || swapIdx == rightGrandchild2);
}

__device__ __inline__ void pushDownKernel(float* dist, int* idx, int m, int size) {
    int cur = m;

    while (true) {
        int swapIdx = cur;

        if (isOnMinLevel(cur)) {
            swapIdx = smallestChildOrGrandchild(dist, cur, size);
            if (swapIdx != -1 && dist[swapIdx] < dist[cur]) {
                swapPair(dist, idx, cur, swapIdx);
                if (isGrandchild(cur, swapIdx) && dist[swapIdx] > dist[PARENT(swapIdx)]) {
                    swapPair(dist, idx, swapIdx, PARENT(swapIdx));
                }
            } else {
                break;
            }
        } else {
            swapIdx = largestChildOrGrandchild(dist, cur, size);
            if (swapIdx != -1 && dist[swapIdx] > dist[cur]) {
                swapPair(dist, idx, cur, swapIdx);
                if (isGrandchild(cur, swapIdx) && dist[swapIdx] < dist[PARENT(swapIdx)]) {
                    swapPair(dist, idx, swapIdx, PARENT(swapIdx));
                }
            } else {
                break;
            }
        }
        cur = swapIdx;
    }
}

__device__ __inline__ void insertKernelDevice(float* d_distances, int* d_indices, int* d_size, float valueDist, int valueIdx, int k) {
    int idx = atomicAdd(d_size, 1);
    d_distances[idx] = valueDist;
    d_indices[idx] = valueIdx;

    int currentIdx = idx;

    while (currentIdx > 0) {
        int parentIdx = PARENT(currentIdx);

        if (isOnMinLevel(currentIdx)) {
            // Min-level logic
            if (d_distances[currentIdx] > d_distances[parentIdx]) {
                swapPair(d_distances, d_indices, currentIdx, parentIdx);
                currentIdx = parentIdx;
                // Push-up max logic
                while (currentIdx > 2 && GRANDPARENT(currentIdx) >= 0 && d_distances[currentIdx] > d_distances[GRANDPARENT(currentIdx)]) {
                    swapPair(d_distances, d_indices, currentIdx, GRANDPARENT(currentIdx));
                    currentIdx = GRANDPARENT(currentIdx);
                }
            } else {
                // Push-up min logic
                while (currentIdx > 2 && GRANDPARENT(currentIdx) >= 0 && d_distances[currentIdx] < d_distances[GRANDPARENT(currentIdx)]) {
                    swapPair(d_distances, d_indices, currentIdx, GRANDPARENT(currentIdx));
                    currentIdx = GRANDPARENT(currentIdx);
                }
                break;
            }
        } else {
            // Max-level logic
            if (d_distances[currentIdx] < d_distances[parentIdx]) {
                swapPair(d_distances, d_indices, currentIdx, parentIdx);
                currentIdx = parentIdx;
                // Push-up min logic
                while (currentIdx > 2 && GRANDPARENT(currentIdx) >= 0 && d_distances[currentIdx] < d_distances[GRANDPARENT(currentIdx)]) {
                    swapPair(d_distances, d_indices, currentIdx, GRANDPARENT(currentIdx));
                    currentIdx = GRANDPARENT(currentIdx);
                }
            } else {
                // Push-up max logic
                while (currentIdx > 2 && GRANDPARENT(currentIdx) >= 0 && d_distances[currentIdx] > d_distances[GRANDPARENT(currentIdx)]) {
                    swapPair(d_distances, d_indices, currentIdx, GRANDPARENT(currentIdx));
                    currentIdx = GRANDPARENT(currentIdx);
                }
                break;
            }
        }
    }
}

__global__ void insertKernel(float* d_distances, int* d_indices, int* d_size, float valueDist, int valueIdx, int k) {
    insertKernelDevice(d_distances, d_indices, d_size, valueDist, valueIdx, k);
    __syncthreads();
}

__global__ void deleteKernel(float* d_distances, int* d_indices, int* d_size, bool isPopMin, float* poppedDistance, int* poppedIndex) {
    int sz = *d_size;
    if (sz == 0) return; 

    float lastDist = d_distances[sz - 1];
    int lastIdx = d_indices[sz - 1];

    if (isPopMin) {
        // Pop Min logic
        *poppedDistance = d_distances[0];
        *poppedIndex = d_indices[0];

        d_distances[0] = lastDist;
        d_indices[0] = lastIdx;
        atomicSub(d_size, 1);
        __syncthreads();
        pushDownKernel(d_distances, d_indices, 0, *d_size);
    } else {
        // Pop Max logic
        int elementsToCopy = (sz > 2) ? 3 : sz;
        float localDist[3];
        // int localIdx[3];

        for (int i = 0; i < elementsToCopy; i++) {
            localDist[i] = d_distances[i];
            // localIdx[i] = d_indices[i];
        }

        int maxIdx = 1;
        if (sz > 2 && localDist[2] > localDist[1]) maxIdx = 2;

        *poppedDistance = d_distances[maxIdx];
        *poppedIndex = d_indices[maxIdx];

        d_distances[maxIdx] = lastDist;
        d_indices[maxIdx] = lastIdx;
        atomicSub(d_size, 1);
        __syncthreads();
        pushDownKernel(d_distances, d_indices, maxIdx, *d_size);
    }
}

#endif // MINMAXHEAP_H
