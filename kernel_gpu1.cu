
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define STARTING_ARRAYSIZE 8
#define BLOCK_DIM 1024
#define CUDA_CALL(x)\
err = x ;\
if(err != cudaSuccess) {\
    printf("Cuda Error: %d in line: %d\n", err, __LINE__);\
}
cudaError_t err;
void findNonzeroRows(Vector *v, CSRMatrix *A) {
    unsigned int nnz = 0;
    for(unsigned int r = 0; r < A->numRows; ++r) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA > 0) {
            if(nnz >= v->capacity) {
                expandVectorCapacity(v, 2 * v->capacity);
            }
            v->data[nnz] = r;
            ++nnz;
        }
    }
    v->nnz = nnz;
}

CSRMatrix *copyCSRToGPU(CSRMatrix *csr) {
    unsigned int *rowPtrs_d, *colIdxs_d;
    float *values_d;
    CSRMatrix *csr_d;
    CUDA_CALL(cudaMalloc((void **)&csr_d, sizeof(CSRMatrix)));
    CUDA_CALL(cudaMalloc((void **)&rowPtrs_d , (csr->numRows + 1) * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&colIdxs_d , csr->capacity * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&values_d , csr->capacity * sizeof(float)));
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(rowPtrs_d, csr->rowPtrs, (csr->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(colIdxs_d, csr->colIdxs, csr->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(values_d, csr->values, csr->capacity * sizeof(float), cudaMemcpyHostToDevice));
    CSRMatrix csr_temp;
    csr_temp.numRows = csr->numRows;
    csr_temp.numCols = csr->numCols;
    csr_temp.nnz = csr->nnz;
    csr_temp.capacity = csr->capacity;
    csr_temp.rowPtrs = rowPtrs_d;
    csr_temp.colIdxs = colIdxs_d;
    csr_temp.values = values_d;
    CUDA_CALL(cudaMemcpy(csr_d, &csr_temp, sizeof(CSRMatrix), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    return csr_d;
}

CSRMatrix *copyCSRFromGPU(CSRMatrix *csr_d) {
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    CUDA_CALL(cudaMemcpy(csr, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    unsigned int *rowPtrs = (unsigned int *)malloc((csr->numRows + 1) * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(rowPtrs, csr->rowPtrs, (csr->numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int *colIdxs = (unsigned int *)malloc(csr->capacity * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(colIdxs, csr->colIdxs, csr->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    float *values = (float *)malloc(csr->capacity * sizeof(float));
    CUDA_CALL(cudaMemcpy(values, csr->values, csr->capacity * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    csr->rowPtrs = rowPtrs;
    csr->colIdxs = colIdxs;
    csr->values = values;
    
    return csr;
}

void freeCSRGPU(CSRMatrix *csr_d) {
    CSRMatrix csr;
    CUDA_CALL(cudaMemcpy(&csr, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaFree(csr.rowPtrs);
    cudaFree(csr.colIdxs);
    cudaFree(csr.values);
    cudaFree(csr_d);
}

CSCMatrix *copyCSCToGPU(CSCMatrix *csc) {
    unsigned int *colPtrs_d, *rowIdxs_d;
    float *values_d;
    CSCMatrix *csc_d;
    CUDA_CALL(cudaMalloc((void **)&csc_d, sizeof(CSCMatrix)));
    CUDA_CALL(cudaMalloc((void **)&colPtrs_d , (csc->numCols + 1) * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&rowIdxs_d , csc->capacity * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&values_d , csc->capacity * sizeof(float)));
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(colPtrs_d, csc->colPtrs, (csc->numCols + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(rowIdxs_d, csc->rowIdxs, csc->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(values_d, csc->values, csc->capacity * sizeof(float), cudaMemcpyHostToDevice));
    CSCMatrix csc_temp;
    csc_temp.numRows = csc->numRows;
    csc_temp.numCols = csc->numCols;
    csc_temp.nnz = csc->nnz;
    csc_temp.capacity = csc->capacity;
    csc_temp.colPtrs = colPtrs_d;
    csc_temp.rowIdxs = rowIdxs_d;
    csc_temp.values = values_d;
    CUDA_CALL(cudaMemcpy(csc_d, &csc_temp, sizeof(CSCMatrix), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    return csc_d;
}

void freeCSCGPU(CSCMatrix *csc_d) {
    CSCMatrix csc;
    CUDA_CALL(cudaMemcpy(&csc, csc_d, sizeof(CSCMatrix), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaFree(csc.colPtrs);
    cudaFree(csc.rowIdxs);  
    cudaFree(csc.values);
    cudaFree(csc_d);
}

__global__ void spmspm_kernel(CSRMatrix *A, CSCMatrix *B, float bias, unsigned int *nnzCounts, unsigned int **colIdxs, float **values) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < A->numRows) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA>0) {
            int currentSize = STARTING_ARRAYSIZE;
            colIdxs[r] = (unsigned int *)malloc(currentSize * sizeof(unsigned int));
            values[r] = (float *)malloc(currentSize * sizeof(float));
            unsigned int* colIdxsA = A->colIdxs + rowPtrA;
            float* valueA = A->values + rowPtrA;
            for(unsigned int c = 0; c < B->numCols; c++) {
                unsigned int colPtrB = B->colPtrs[c];
                unsigned int nnzB = B->colPtrs[c + 1] - colPtrB;
                if(nnzB>0) {
                    unsigned int* rowIdxsB = B->rowIdxs + colPtrB;
                    float* valueB = B->values + colPtrB;
                    // Loop and find intersection
                    float sum = 0.0f;
                    unsigned int ia = 0, ib = 0;
                    while(ia < nnzA && ib < nnzB) {
                        unsigned int colIdx = colIdxsA[ia];
                        unsigned int rowIdx = rowIdxsB[ib];
                        if(colIdx < rowIdx) {
                            ia++;
                        } else if(colIdx > rowIdx) {
                            ib++;
                        } else {
                            sum += valueA[ia]*valueB[ib];
                            ia++;
                            ib++;
                        }
                    }

                    if(sum > THRESHOLD || sum < -THRESHOLD) {
                        sum += bias;
                        //Remove negative and zero values
                        if(sum > 0) {
                            if(sum>YMAX) {
                                sum = YMAX;
                            }
                            if(nnzCounts[r] == currentSize) {
                                currentSize += STARTING_ARRAYSIZE;
                                unsigned int *colIdxs_d = (unsigned int *)malloc(currentSize * sizeof(unsigned int));
                                float *values_d = (float *)malloc(currentSize * sizeof(float));
                                memcpy(colIdxs_d, colIdxs[r], nnzCounts[r] * sizeof(unsigned int));
                                memcpy(values_d, values[r], nnzCounts[r] * sizeof(float));
                                free(colIdxs[r]);
                                free(values[r]);
                                colIdxs[r] = colIdxs_d;
                                values[r] = values_d;
                            }
                            colIdxs[r][nnzCounts[r]] = c;
                            values[r][nnzCounts[r]] = sum;
                            nnzCounts[r]++;
                        }    
                    }
                }
            }
        }
    }
}

__global__ void scan_kernel(unsigned int *input, unsigned int *output, unsigned int size) {
    __shared__ unsigned int buffer1_s[BLOCK_DIM]; 
    __shared__ unsigned int buffer2_s[BLOCK_DIM];
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int *inBuffer_s = buffer1_s;
    unsigned int *outBuffer_s = buffer2_s; 

    if(i < size) {
        if(threadIdx.x == 0) {
            inBuffer_s[threadIdx.x] = 0.0f; } 
        else {
            inBuffer_s[threadIdx.x] = input[i - 1]; 
        }
    
        __syncthreads();
    
        for(unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2) { 
            if(threadIdx.x >= stride) {
                outBuffer_s[threadIdx.x] =
                inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
            } else {
                outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
            }
            __syncthreads();
            unsigned int *tmp = inBuffer_s; 
            inBuffer_s = outBuffer_s; 
            outBuffer_s = tmp;
        }
    
        output[i] = inBuffer_s[threadIdx.x];
    }
}

__global__ void reAssemble_kernel(unsigned int* nnzCounts, unsigned int numRows, unsigned int** colIdxs, float** values, unsigned int* colIdxsRes, float* valuesRes) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(r < numRows) {
        unsigned int nnz = nnzCounts[r + 1] - nnzCounts[r];
        
        if(nnz > 0) {
            memcpy(colIdxsRes + r, colIdxs[r], nnz * sizeof(unsigned int));
            memcpy(valuesRes + r, values[r], nnz * sizeof(unsigned int));
            free(colIdxs[r]);
            free(values[r]);
        }
    }
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1536*1024*1024);
    Timer timer;

    // Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix *Y0 = createCSRfromCOO(featureVectors);
    unsigned int numRows = Y0->numRows;
    CSRMatrix *Y0_d = copyCSRToGPU(Y0);
    freeCSR(Y0);
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");
    unsigned int** colIdxs;
    CUDA_CALL(cudaMalloc((void **)&colIdxs, numRows * sizeof(unsigned int *)));
    float** values;
    CUDA_CALL(cudaMalloc((void **)&values, numRows * sizeof(float *)));
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (numRows + numThreadsPerBlock - 1) / numThreadsPerBlock;
    CSRMatrix *inBuffer = Y0_d;
    CSRMatrix reAssembledSCR;
    reAssembledSCR.numRows = numRows;
    
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        CUDA_CALL(cudaMalloc((void **)&reAssembledSCR.rowPtrs, (numRows + 1) * sizeof(unsigned int)));
        cudaDeviceSynchronize();
        CUDA_CALL(cudaMemset(reAssembledSCR.rowPtrs, 0, (numRows + 1) * sizeof(unsigned int)));
        CSCMatrix *W = createCSCfromCOO(layerWeights[layer]);
        CSCMatrix *W_d = copyCSCToGPU(W);
        reAssembledSCR.numCols = W->numCols;
        freeCSC(W);
        spmspm_kernel<<< numBlocks, numThreadsPerBlock >>>(inBuffer, W_d, bias, reAssembledSCR.rowPtrs, colIdxs, values);
        cudaDeviceSynchronize();
        freeCSCGPU(W_d);
        scan_kernel<<< numBlocks, numThreadsPerBlock >>>(reAssembledSCR.rowPtrs, reAssembledSCR.rowPtrs, numRows + 1);
        cudaDeviceSynchronize();
        cudaFree(reAssembledSCR.colIdxs);
        cudaFree(reAssembledSCR.values);
        unsigned int nnz = 0;
        cudaMemcpy(&nnz, reAssembledSCR.rowPtrs + numRows, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf(" nnz = %d", nnz);
        reAssembledSCR.nnz = nnz;
        reAssembledSCR.capacity = nnz;
        CUDA_CALL(cudaMalloc((void **)&reAssembledSCR.colIdxs, nnz * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc((void **)&reAssembledSCR.values, nnz * sizeof(float)));
        cudaDeviceSynchronize();
        reAssemble_kernel<<< numBlocks, numThreadsPerBlock >>>(reAssembledSCR.rowPtrs, numRows, colIdxs, values, reAssembledSCR.colIdxs, reAssembledSCR.values);
        freeCSRGPU(inBuffer);
        CUDA_CALL(cudaMemcpy(inBuffer, &reAssembledSCR, sizeof(CSRMatrix), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "");
    }

    // Find nonzero rows
    startTime(&timer);
    CSRMatrix *YFinal = copyCSRFromGPU(inBuffer);
    freeCSRGPU(inBuffer);
    findNonzeroRows(result, YFinal);
    stopTimeAndPrint(&timer, "Find nonzero rows");
}
