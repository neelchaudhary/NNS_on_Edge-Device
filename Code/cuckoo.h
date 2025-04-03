#ifndef CUCKOO_H
#define CUCKOO_H

#include <iostream>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>

#define S 4  // fingerprints per bucket

#include "cuckoo.h"

__device__ __inline__ uint32_t murmurHash64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x; 
}

__device__ __inline__ uint32_t xorshiftHash64(uint64_t x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return (uint32_t)x;
}

__device__ __inline__ uint16_t fingerprintFromHash(uint32_t h) {
    return (uint16_t)(h & 0xFFFF); 
}

// Attempts to insert a fingerprint into a given bucket index
__device__ __inline__ bool tryInsert(uint64_t* buckets, uint32_t B, uint16_t fingerprint, uint32_t index) {
    uint64_t oldVal = atomicAdd((unsigned long long*)&buckets[index], 0ULL); // read current value
    for (int slot = 0; slot < S; slot++) {
        int shift = slot * 16;
        uint16_t slot_val = (uint16_t)((oldVal >> shift) & 0xFFFF);
        if (slot_val == 0) {
            // Free slot found
            uint64_t mask = ~(0xFFFFULL << shift);
            uint64_t newVal = (oldVal & mask) | (((uint64_t)fingerprint) << shift);
            uint64_t prev = atomicCAS((unsigned long long*)&buckets[index], oldVal, newVal);
            if (prev == oldVal) {
                return true; // successfully inserted
            }
            // CAS failed, try again from the start
            oldVal = atomicAdd((unsigned long long*)&buckets[index], 0ULL); 
            slot = -1; 
        }
    }
    return false; 
}

__device__ __inline__ bool cuckooLookup(uint64_t* buckets, uint32_t B, uint64_t element) {
    uint32_t h = murmurHash64(element);
    uint16_t fp = fingerprintFromHash(h);
    uint32_t h2 = murmurHash64((uint64_t)fp);

    uint32_t index1 = h % B;
    uint32_t index2 = (index1 ^ h2) % B;

    // Lambda to check a bucket
    auto checkBucket = [&](uint32_t idx) {
        uint64_t val = atomicAdd((unsigned long long*)&buckets[idx], 0ULL); 
        for (int i = 0; i < S; i++) {
            int shift = i * 16;
            uint16_t slot_val = (uint16_t)((val >> shift) & 0xFFFF);
            if (slot_val == fp) return true;
        }
        return false;
    };

    return checkBucket(index1) || checkBucket(index2);
}


__device__ __inline__ void cuckooInsertKernel(uint64_t* buckets, uint32_t B, const uint64_t* elements, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint64_t element = elements[tid];
    uint32_t h = murmurHash64(element);
    uint16_t fp = fingerprintFromHash(h);

    uint32_t h2 = murmurHash64((uint64_t)fp);
    uint32_t index1 = h % B;
    uint32_t index2 = (index1 ^ h2) % B;

    bool inserted = tryInsert(buckets, B, fp, index1);
    if (!inserted) {
        inserted = tryInsert(buckets, B, fp, index2);
    }

    // Kick-out loop if needed
    int maxAttempts = 500;
    uint16_t curFp = fp;
    uint32_t curIndex = inserted ? 0 : index1;

    // Only proceed if not inserted yet
    while (!inserted && maxAttempts-- > 0) {
        // Read the current bucket
        uint64_t oldVal = atomicAdd((unsigned long long*)&buckets[curIndex], (unsigned long long)0);

        // Choose a slot to kick out
        // This can be any slot; use h again for pseudo-randomness
        int slot = h % S;
        int shift = slot * 16;
        uint16_t victimFp = (uint16_t)((oldVal >> shift) & 0xFFFF);

        if (victimFp == 0) {
            // There's a free slot after all
            uint64_t mask = ~(0xFFFFULL << shift);
            uint64_t newVal = (oldVal & mask) | (((uint64_t)curFp) << shift);
            uint64_t prev = atomicCAS((unsigned long long*)&buckets[curIndex], oldVal, newVal);
            if (prev == oldVal) {
                inserted = true;
            }
            // If failed, loop continues
            continue;
        }

        // Replace victim with current fp
        uint64_t mask = ~(0xFFFFULL << shift);
        uint64_t newVal = (oldVal & mask) | (((uint64_t)curFp) << shift);
        uint64_t prev = atomicCAS((unsigned long long*)&buckets[curIndex], oldVal, newVal);
        if (prev == oldVal) {
            // Successfully placed curFp, now relocate victimFp
            uint32_t vh = murmurHash64((uint64_t)victimFp);
            uint32_t victimAlt = (curIndex ^ vh) % B;
            curFp = victimFp;
            curIndex = victimAlt;

            if (tryInsert(buckets, B, curFp, curIndex)) {
                inserted = true;
            }
        }
    }
}

__global__ void cuckooLookupKernel(uint64_t* buckets, uint32_t B, const uint64_t* queries, int q, int* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < q) {
        uint64_t element = queries[tid];
        results[tid] = cuckooLookup(buckets, B, element) ? 1 : 0;
    }
}
#endif // CUCKOO_H
