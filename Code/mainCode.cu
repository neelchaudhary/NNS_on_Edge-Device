#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "minMaxHeap.h"
#include "cuckoo.h"
#include "heap.h"


using namespace std;

#define CHECK_CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
      exit(code);
   }
}

// Reads an fvecs file into a 2D vector<float> format
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

std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<int>> data;
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (true) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), 4);
        if (!file.good()) break;
        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!file.good()) break;
        data.push_back(vec);
    }
    file.close();
    return data;
}

// Flatten 2D vector into 1D
std::vector<float> flatten(const std::vector<std::vector<float>>& data) {
    std::vector<float> flat_data;
    for (auto &v : data) {
        flat_data.insert(flat_data.end(), v.begin(), v.end());
    }
    return flat_data;
}

__global__ void computeDistancesForNeighbors(
    const float* __restrict__ d_data,
    const float* __restrict__ d_query,
    const int* __restrict__ d_adjacency,
    int nbr_start, int nbr_count, int d,
    float* __restrict__ d_nbr_distances,
    int* __restrict__ d_nbr_indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbr_count) return;
    int nbr_idx = d_adjacency[nbr_start + i];
    // if (nbr_idx < 0 || nbr_idx >= n) return;

    float dist = 0.0f;
    const float* vec = &d_data[nbr_idx * d];
    for (int dim = 0; dim < d; dim++) {
        float diff = vec[dim] - d_query[dim];
        dist += diff * diff;
    }

    d_nbr_distances[i] = dist;
    d_nbr_indices[i] = nbr_idx;
}

__global__ void processCandidateKernel(
    float* d_poppedDist, int* d_poppedIdx,
    float* d_topk_dist, int* d_topk_idx, int k, int* d_topk_size,
    bool* d_terminateFlag)
{
    float dist = *d_poppedDist;
    int idx = *d_poppedIdx;

    // Update top-k
    heap_insert(d_topk_dist, d_topk_idx, k, dist, idx, d_topk_size);

    // For now, do not terminate early:
    *d_terminateFlag = false;
}

__global__ void checkVisitedAndMarkKernel(
    int popped_idx,
    bool* d_alreadyVisited,
    uint64_t* d_buckets, uint32_t B, int n
    // Add cuckoo filter arrays if needed (e.g. uint64_t* d_buckets, uint32_t B)
) {
    // If cuckoo_contains(...) is implemented:
    if (cuckooLookup(d_buckets, B, (uint64_t)popped_idx)) {
        *d_alreadyVisited = true;
    } else {
        uint64_t element = static_cast<uint64_t>(popped_idx);
        cuckooInsertKernel(d_buckets, B, &element, n);
        *d_alreadyVisited = false;
    }

    *d_alreadyVisited = false; // Placeholder logic
}

__global__ void insertNeighborsKernel(
    float* d_nbr_distances, int* d_nbr_indices, int nbr_count,
    float* d_mm_distances, int* d_mm_indices, int* d_mm_size,
    float* d_topk_dist, int* d_topk_idx, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbr_count) return;

    float dist = d_nbr_distances[i];
    int idx = d_nbr_indices[i];

    // If dist improves top-k:
    if (dist < d_topk_dist[k-1]) {
        // Insert into min-max heap using device_insertMinMax:
        insertKernelDevice(d_mm_distances, d_mm_indices, d_mm_size, dist, idx, k);
    }
}

__global__ void computeInitialDistances(
    const float* __restrict__ d_data, 
    const float* __restrict__ d_queries, 
    float* __restrict__ d_initial_distances, 
    int n_queries, int d) 
{
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= n_queries) return;

    float dist = 0.0f;
    const float* query_vec = &d_queries[query_idx * d];
    const float* start_vec = d_data;

    for (int dim = 0; dim < d; dim++) {
        float diff = query_vec[dim] - start_vec[dim];
        dist += diff * diff;
    }

    d_initial_distances[query_idx] = dist;
}





int main() {
    // Adjust these parameters as needed
    std::string baseFile = "sift/sift_base.fvecs";
    std::string queryFile = "sift/sift_query.fvecs";
    std::string groundTruthFile = "sift/sift_groundtruth.ivecs";
    std::string neighbors_file = "sift/neighbors.ivecs";
    int no_of_neighbors = 100;           // number of neighbors
    int n = 1000000;         // number of vectors (use smaller subset for test)
    int d = 128;           // dimension
    // int capacity = 10;
    int k = 100;
    
    // Load data from file
    std::vector<std::vector<float>> data = read_fvecs(baseFile);
    if ((int)data.size() < n) {
        std::cerr << "File has fewer than n vectors. Using " << data.size() << " instead of " << n << ".\n";
        n = (int)data.size();
    }
    std::vector<float> flat_data = flatten(data);

    auto neighbors = read_ivecs(neighbors_file);
    int nb_dim = (int)neighbors[0].size(); // likely 100
    // Flatten the 2D vector to a 1D vector first
    std::vector<int> flat_neighbors;
    for (const auto& neighbor_row : neighbors) {
        flat_neighbors.insert(flat_neighbors.end(), neighbor_row.begin(), neighbor_row.end());
    }

    int *d_neighbors;
    CHECK_CUDA_ERR(cudaMalloc(&d_neighbors, n * nb_dim * sizeof(int)));
    CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, flat_neighbors.data(), n * nb_dim * sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "Number of neighbors per node is : " << nb_dim << std::endl;

    // Allocate device memory
    float *d_data;
    CHECK_CUDA_ERR(cudaMalloc(&d_data, n * d * sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_data, flat_data.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));

    auto query_data = read_fvecs(queryFile);
    int num_queries = (int)query_data.size();
    std::vector<float> flat_queries = flatten(query_data);
    int n_queries = num_queries;
    int dim = d;
    float *d_queries;
    CHECK_CUDA_ERR(cudaMalloc(&d_queries, num_queries*d*sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_queries, flat_queries.data(), num_queries*d*sizeof(float), cudaMemcpyHostToDevice));
    
    float* d_initial_distances;
    cudaMalloc(&d_initial_distances, n_queries * sizeof(float));

    // Launch the kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (n_queries + threadsPerBlock - 1) / threadsPerBlock;
    computeInitialDistances<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_queries, d_initial_distances, n_queries, dim);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERR(cudaGetLastError());

    float *d_query;
    CHECK_CUDA_ERR(cudaMalloc(&d_query, d*sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_query, flat_queries.data(), d*sizeof(float), cudaMemcpyHostToDevice));
    float first_query_distance;
    // Copy the first element from the device array to the host
    CHECK_CUDA_ERR(cudaMemcpy(&first_query_distance, d_initial_distances, sizeof(float), cudaMemcpyDeviceToHost));

    // Flatten data
    auto groundtruth_data = read_ivecs(groundTruthFile);
    int gt_dim = (int)groundtruth_data[0].size(); // likely 100
    std::cout << "Number of queries: " << num_queries << ", Ground truth neighbors per query: " << gt_dim << std::endl;
    int *d_groundtruth;
    CHECK_CUDA_ERR(cudaMalloc(&d_groundtruth, num_queries*gt_dim*sizeof(int)));
    CHECK_CUDA_ERR(cudaMemcpy(d_groundtruth, groundtruth_data.data(), num_queries*gt_dim*sizeof(int), cudaMemcpyHostToDevice));

    //Min-Max Heap 
    float *d_mm_distances;
    int *d_mm_indices, *d_mm_size, *d_mm_capacity;
    cudaMalloc(&d_mm_distances, k*sizeof(float));
    cudaMalloc(&d_mm_indices, k*sizeof(int));
    cudaMalloc(&d_mm_size, sizeof(int));
    cudaMalloc(&d_mm_capacity, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_mm_size, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mm_capacity, &k, sizeof(int), cudaMemcpyHostToDevice);

    // Allocate top-k arrays:
    float d_topk_dist[k];
    int d_topk_idx[k];
    for (int i = 0; i < k; i++) {
       d_topk_dist[i] = 1e9f;
       d_topk_idx[i] = -1;
    }

    //top k max heap
    float *d_topk_dist_dev;
    int *d_topk_idx_dev, *d_topk_size;
    cudaMalloc(&d_topk_dist_dev, k*sizeof(float));
    cudaMalloc(&d_topk_idx_dev, k*sizeof(int));
    cudaMalloc(&d_topk_size, sizeof(int));
    cudaMemcpy(d_topk_dist_dev, d_topk_dist, k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topk_idx_dev, d_topk_idx, k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topk_size, &zero, sizeof(int), cudaMemcpyHostToDevice);


    //Cuckoo Kernel
    size_t B = 1000000;     // number of buckets
    // size_t total_capacity = B;
    uint64_t* d_buckets;
    cudaMalloc(&d_buckets, B * sizeof(uint64_t));
    cudaMemset(d_buckets, 0, B * sizeof(uint64_t));

    float *d_poppedDist;
    int *d_poppedIdx;
    bool *d_terminateFlag, *d_alreadyVisited;
    cudaMalloc(&d_poppedDist, sizeof(float));
    cudaMalloc(&d_poppedIdx, sizeof(int));
    cudaMalloc(&d_terminateFlag, sizeof(bool));
    cudaMalloc(&d_alreadyVisited, sizeof(bool));

    // Insert start node into min-max heap:
    int start_idx = 0;
    insertKernel<<<1,1>>>(d_mm_distances, d_mm_indices, d_mm_size, first_query_distance, start_idx, k);
    // uint64_t element = static_cast<uint64_t>(0);
    cudaDeviceSynchronize(); 
    CHECK_CUDA_ERR(cudaGetLastError());

    bool host_terminate = false;

    while(!host_terminate) {
        bool isPopMin = true;
        // Pop best candidate
        deleteKernel<<<1,1>>>(d_mm_distances, d_mm_indices, d_mm_size, isPopMin, d_poppedDist, d_poppedIdx);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());

        // Process candidate
        processCandidateKernel<<<1,1>>>(d_poppedDist, d_poppedIdx, d_topk_dist_dev, d_topk_idx_dev, k, d_topk_size, d_terminateFlag);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());

        bool h_terminateFlag;
        cudaMemcpy(&h_terminateFlag, d_terminateFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_terminateFlag) break;

        int h_poppedIdx;
        float h_poppedDistVal;
        cudaMemcpy(&h_poppedIdx, d_poppedIdx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_poppedDistVal, d_poppedDist, sizeof(float), cudaMemcpyDeviceToHost);

        // Check visited
        checkVisitedAndMarkKernel<<<1,1>>>(h_poppedIdx, d_alreadyVisited, d_buckets, (uint32_t)B, n);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());

        bool h_alreadyVisited;
        cudaMemcpy(&h_alreadyVisited, d_alreadyVisited, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_alreadyVisited) {
            int h_mm_size;
            cudaMemcpy(&h_mm_size, d_mm_size, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_mm_size == 0) break;
            continue;
        }

        int nbr_count = no_of_neighbors; // Fixed number of neighbors per query

        computeDistancesForNeighbors<<<(nbr_count + 255) / 256, 256>>>(
            d_data, d_query, d_neighbors + (h_poppedIdx * no_of_neighbors),
            0, nbr_count, d,
            d_mm_distances, d_mm_indices);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());

        insertNeighborsKernel<<<(nbr_count + 255) / 256, 256>>>(
            d_mm_distances, d_mm_indices, nbr_count,
            d_mm_distances, d_mm_indices, d_mm_size,
            d_topk_dist_dev, d_topk_idx_dev, k);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());

        int h_mm_size;
        cudaMemcpy(&h_mm_size, d_mm_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_mm_size == 0) break;
    }

    // After done
    std::vector<float> host_topk_dist(k);
    std::vector<int> host_topk_idx(k);
    cudaMemcpy(host_topk_dist.data(), d_topk_dist_dev, k*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_topk_idx.data(), d_topk_idx_dev, k*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Top-k results:\n";
    for (int i = 0; i < k; i++) {
        std::cout << i << ": idx=" << host_topk_idx[i] << ", dist=" << host_topk_dist[i] << "\n";
    }
    std::cout << "\nGroundtruth: ";
    for (int i = 0; i < k; i++){
        std::cout << groundtruth_data[0][i] << " ";
    }
    std::cout<<endl;

    // Free all CUDA allocations
    cudaFree(d_data);
    cudaFree(d_queries);
    cudaFree(d_query);
    cudaFree(d_neighbors);
    cudaFree(d_groundtruth);
    cudaFree(d_initial_distances);

    // Min-Max Heap allocations
    cudaFree(d_mm_distances);
    cudaFree(d_mm_indices);
    cudaFree(d_mm_size);
    cudaFree(d_mm_capacity);

    // Top-k heap allocations
    cudaFree(d_topk_dist_dev);
    cudaFree(d_topk_idx_dev);
    cudaFree(d_topk_size);

    // Cuckoo filter allocations
    cudaFree(d_buckets);

    // Other temporary allocations
    cudaFree(d_poppedDist);
    cudaFree(d_poppedIdx);
    cudaFree(d_terminateFlag);
    cudaFree(d_alreadyVisited);
    cudaDeviceReset();

    return 0;
}
