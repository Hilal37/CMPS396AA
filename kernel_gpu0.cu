
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define BLOCK_DIM 1024

__global__ void spmspm(COOMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < A->numRows) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA>0) {
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
                            int nnzIdx = atomicAdd(&result->nnz, 1);
                            result->rowIdxs[nnzIdx] = r;
                            result->colIdxs[nnzIdx] = c;
                            result->values[nnzIdx] = sum;
                        }    
                    }
                }
            }
        }
    }
}

void findNonzeroRows(Vector* v, CSRMatrix* A) {
    unsigned int nnz = 0;
    for(unsigned int r = 0; r < A->numRows; ++r) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA > 0) {
            if(nnz >= v->capacity) {
                expandVectorCapacity(v, 2*v->capacity);
            }
            v->data[nnz] = r;
            ++nnz;
        }
    }
    v->nnz = nnz;
}

CSCMatrix* copyCSCToGPU(CSCMatrix* csc) {
    CSCMatrix* csc_d;
    cudaMalloc((void **)&csc_d , sizeof(CSRMatrix));
    cudaMemcpy(csc_d, csc, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csc_d->colPtrs , (csc->numCols + 1) * sizeof(unsigned int));
    cudaMemcpy(csc_d->colPtrs, csc->colPtrs, (csc->numCols + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(csc_d->colPtrs, 0, (csc->numCols + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&csc_d->rowIdxs , csc->nnz * sizeof(unsigned int));
    cudaMemcpy(csc_d->rowIdxs, csc->rowIdxs, csc->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csc_d->values , csc->nnz * sizeof(float));
    cudaMemcpy(csc_d->values, csc->values, csc->nnz * sizeof(float), cudaMemcpyHostToDevice);
    
    return csc_d;
}

CSRMatrix* copyCSRToGPU(CSRMatrix* csr) {
    CSRMatrix* csr_d;
    cudaMalloc((void **)&csr_d , sizeof(CSRMatrix));
    cudaMemcpy(csr_d, csr, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csr_d->rowPtrs , (csr->numRows + 1) * sizeof(unsigned int));
    cudaMemcpy(csr_d->rowPtrs, csr->rowPtrs, (csr->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csr_d->colIdxs , csr->nnz * sizeof(unsigned int));
    cudaMemcpy(csr_d->colIdxs, csr->colIdxs, csr->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csr_d->values , csr->nnz * sizeof(float));
    cudaMemcpy(csr_d->values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice);
    
    return csr_d;
}

COOMatrix* copyCOOToGPU(COOMatrix* coo) {
    COOMatrix* coo_d;
    cudaMalloc((void **)&coo_d , sizeof(COOMatrix));
    cudaMemcpy(coo_d, coo, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&coo_d->rowIdxs , coo->capacity * sizeof(unsigned int));
    cudaMemcpy(coo_d->rowIdxs, coo->rowIdxs, coo->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&coo_d->colIdxs , coo->capacity * sizeof(unsigned int));
    cudaMemcpy(coo_d->colIdxs, coo->colIdxs, coo->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&coo_d->values , coo->capacity * sizeof(float));
    cudaMemcpy(coo_d->values, coo->values, coo->capacity * sizeof(float), cudaMemcpyHostToDevice);
    
    return coo_d;
}

void freeCSRGPU(CSRMatrix* csr) {
    cudaFree(csr->rowPtrs);
    cudaFree(csr->colIdxs);
    cudaFree(csr->values);
    cudaFree(csr);
}

void freeCSCGPU(CSCMatrix* csc) {
    cudaFree(csc->colPtrs);
    cudaFree(csc->rowIdxs);  
    cudaFree(csc->values);
    cudaFree(csc);
}

void freeCOOGPU(COOMatrix* coo) {
    cudaFree(coo->rowIdxs); 
    cudaFree(coo->colIdxs); 
    cudaFree(coo->values);
    cudaFree(coo);
}

COOMatrix* createEmptyCOO_gpu(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix* coo = (COOMatrix*) malloc(sizeof(COOMatrix));
    coo->numRows = numRows;
    coo->numCols = numCols;
    coo->nnz = 0;
    coo->capacity = capacity;
    coo->rowIdxs = (unsigned int *)malloc(capacity*sizeof(unsigned int));
    coo->colIdxs = (unsigned int *)malloc(capacity*sizeof(unsigned int));
    coo->values = (float *)malloc(capacity*sizeof(float));
    COOMatrix *coo_d = copyCOOToGPU(coo);
    freeCOO(coo);
    
    return coo_d;
}


//custom code: create CSR form COO in parallel

//parallel function for binning
__global__ void binning_kernel(unsigned int* rowPtrs, unsigned int* out_colIdxs, float* out_values, unsigned int* in_rowIdxs, unsigned int* in_colIdxs, float* in_values, unsigned int size) {
    //filling the CSR's colIdxs and values (out_colIdxs and out_values) using the temp rowPtrs
    //NOTE: here rowPtrs is just a temp copy of the actual rowPtrs, so NO NEED TO RESTORE PTRS

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int row = in_rowIdxs[idx];
    unsigned int i = rowPtrs[row]++;


    out_colIdxs[i] = in_colIdxs[idx];
    out_values[i] = in_values[idx];
}

//parallel function to perform scan
__global__ void scan_kernel(unsigned int *input, unsigned int *output) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ unsigned int buffer1_s[BLOCK_DIM]; 
    __shared__ unsigned int buffer2_s[BLOCK_DIM]; 
    unsigned int* inBuffer_s = buffer1_s;
    unsigned int* outBuffer_s = buffer2_s; 
    
    if(threadIdx.x == 0) {
        inBuffer_s[threadIdx.x] = 0.0f; } 
    else {
        inBuffer_s[threadIdx.x] = input[i - 1]; 
    }

    __syncthreads();

    for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) { 
        if(threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] =
            inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        unsigned int* tmp = inBuffer_s; 
        inBuffer_s = outBuffer_s; 
        outBuffer_s = tmp;
    }
    
    // if(threadIdx.x == BLOCK_DIM - 1) { 
    //     partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    // }
    output[i] = inBuffer_s[threadIdx.x];
}

//parallel function to compute histogram (adapted from asst. 6)
__global__ void histogram_kernel(float* values, unsigned int* bins, unsigned int size, unsigned int num_bins) {

    extern __shared__ int private_bins[];

    //initializing private bins to zero
    //(for loop is just in case block dim is smaller than the number of bins, 
    //so each thread in the block has to initialize more than one bin to zero)

    for(unsigned int i = 0; i < (num_bins + blockDim.x - 1)/blockDim.x; ++i) {
        if(i*blockDim.x + threadIdx.x < num_bins) {
            private_bins[i*blockDim.x + threadIdx.x] = 0;
        }
    }

    __syncthreads();

    //updating private bins
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < size) {
        unsigned char b = values[idx];
        atomicAdd(&private_bins[b], 1);
    }

    __syncthreads();

    //committing changes to global bins
    //for loop: same logic as the initialization to zero
    for(unsigned int i = 0; i < (num_bins + blockDim.x - 1)/blockDim.x; ++i) {
        if(i*blockDim.x + threadIdx.x < num_bins && private_bins[i*blockDim.x + threadIdx.x] != 0) {
            atomicAdd(&bins[i*blockDim.x + threadIdx.x], private_bins[i*blockDim.x + threadIdx.x]);
        }
    }
}

CSRMatrix* createCSRfromCOO_gpu(COOMatrix* A) {
    //step 1: allocate arrays
    //output arrays
    unsigned int* rowPtrs, *colIdxs;
    float* values;

    cudaMalloc((void**) &rowPtrs, (A->numRows + 1) * sizeof(unsigned int));
    cudaMemset(rowPtrs, 0, (A->numRows + 1) * sizeof(unsigned int)); //initialize all rowPtrs to zero
    cudaMalloc((void**) &colIdxs, A->nnz * sizeof(unsigned int));
    cudaMalloc((void**) &values, A->nnz * sizeof(float));

    //copying A->rowIdxs, A->colIdxs, A->values to GPU
    unsigned int* rowIdxs_A, *colIdxs_A;
    float* values_A;
    cudaMalloc((void**) &rowIdxs_A, A->nnz*sizeof(unsigned int));
    cudaMalloc((void**) &colIdxs_A, A->nnz*sizeof(unsigned int));
    cudaMalloc((void**) &values_A, A->nnz*sizeof(float));
    cudaMemcpy(rowIdxs_A, A->rowIdxs, A->nnz*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIdxs_A, A->colIdxs, A->nnz*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_A, A->values, A->nnz*sizeof(float), cudaMemcpyHostToDevice);

    //cudaDeviceSynchronize();

    //Now we need to compute the rowPtrs (steps 2 and 3)
    //step 2: Histogram (how many non-zeros for each row)
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (A->nnz + numThreadsPerBlock - 1)/numThreadsPerBlock; 
    histogram_kernel <<< numBlocks, numThreadsPerBlock, (A->numRows + 1) * sizeof(int)>>>(values_A, rowPtrs, A->nnz, A->numRows + 1);
    cudaDeviceSynchronize();

    //step 3: prefix sum on the rowPtrs
    unsigned int numBlocksScan = (A->numRows + numThreadsPerBlock)/numThreadsPerBlock; //(for the ceiling, + 1 - 1 cancel out)
    scan_kernel<<< numBlocksScan, numThreadsPerBlock >>>(rowPtrs, rowPtrs);
    cudaDeviceSynchronize();

    //step 4: binning (populating the colIdxs and values arrays)
    //creating a temp copy of rowPtrs
    unsigned int* rowPtrs_temp;
    cudaMalloc(&rowPtrs_temp, (A->numRows + 1)*sizeof(unsigned int));
    cudaMemcpy(rowPtrs_temp, rowPtrs, (A->numRows + 1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    binning_kernel <<< numBlocks, numThreadsPerBlock >>>(rowPtrs_temp, colIdxs, values, rowIdxs_A, colIdxs_A, values_A, A->nnz);
    cudaDeviceSynchronize();

    //copy rowPtrs, colIdxs, and values to CPU (_h for host)
    unsigned int* rowPtrs_h = (unsigned int *) calloc(A->numRows + 1, sizeof(unsigned int));
    unsigned int* colIdxs_h = (unsigned int *) malloc( A->nnz * sizeof(unsigned int));
    float* values_h = (float *)malloc( A->nnz * sizeof(float));

    cudaMemcpy(rowPtrs_h, rowPtrs, (A->numRows + 1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(colIdxs_h, colIdxs, A->nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values_h, values, A->nnz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //now that we have all the pieces needed, build the CSR and return it
    CSRMatrix* csr = (CSRMatrix*) malloc(sizeof(CSRMatrix));
    csr->numRows = A->numRows;
    csr->numCols = A->numCols;
    csr->nnz = A->nnz;
    csr->capacity = A->nnz;
    csr->rowPtrs = rowPtrs_h;
    csr->colIdxs = colIdxs_h;
    csr->values = values_h;

    return csr;
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    Timer timer;

    // Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix* Y0 = createCSRfromCOO(featureVectors);
    CSRMatrix* Y0_d = copyCSRToGPU(Y0);
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    CSCMatrix* W_d[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCfromCOO(layerWeights[layer]);
        W_d[layer] = copyCSCToGPU(W[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");

    // Double buffers
    startTime(&timer);
    CSRMatrix *tmp = createEmptyCSR(Y0->numRows, Y0->numCols, 2*Y0->nnz);
    CSRMatrix *tmp_d = copyCSRToGPU(tmp);
    CSRMatrix *inBuffer  = Y0_d;
    CSRMatrix *outBuffer = tmp_d;
    stopTimeAndPrint(&timer, "Allocate temporary buffer");
    unsigned int numThreadsPerBlock = 1024;
        
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        unsigned int numBlocks = (inBuffer->numRows + numThreadsPerBlock - 1)/numThreadsPerBlock; 

        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        COOMatrix* res = createEmptyCOO_gpu(inBuffer->numRows, outBuffer->numCols, inBuffer->numRows * outBuffer->numCols);
        spmspm<<< numBlocks, numThreadsPerBlock >>>(res, inBuffer, W_d[layer], bias);
        stopTimeAndPrint(&timer, "");
        outBuffer = createCSRfromCOO_gpu(res);
        freeCOOGPU(res);
        // Swap buffers
        CSRMatrix *t = inBuffer;
        inBuffer = outBuffer;
        outBuffer = t;

    }

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, inBuffer);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSR(Y0);
    freeCSRGPU(Y0_d);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSC(W[layer]);
        freeCSCGPU(W_d[layer]);
    }
    freeCSR(tmp);
    freeCSRGPU(tmp_d);
    stopTimeAndPrint(&timer, "Deallocate memory");

}