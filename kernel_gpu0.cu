
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

__global__ void spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {
    unsigned int nnzIdx= 0;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if(r < A->numRows) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA>0) {
            unsigned int* colIdxsA = A->colIdxs + rowPtrA;
            float* valueA = A->values + rowPtrA;
            int c = blockIdx.x * blockDim.x + threadIdx.x;
            if(c < B->numCols) {
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
                            if(nnzIdx >= result->capacity) {
                                expandCSRCapacity(result, 2*result->capacity);
                            }
                            result->colIdxs[nnzIdx] = c;
                            result->values[nnzIdx] = sum;
                            ++nnzIdx;
                        }    
                    }
                }
            }
        }
        result->rowPtrs[r + 1] = nnzIdx;
    }
    result->nnz = nnzIdx;
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
}

CSCMatrix* copyCSRToGPU(CSRMatrix* csr) {
    CSCMatrix* csr_d;
    cudaMalloc((void **)&csr_d , sizeof(CSRMatrix));
    cudaMemcpy(csr_d, csr, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csr_d->rowPtrs , (csr->numRows + 1) * sizeof(unsigned int));
    cudaMemcpy(csr_d->rowPtrs, csr->rowPtrs, (csr->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(csr_d->rowPtrs, 0, (csr->numRows + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&csr_d->colIdxs , csr->nnz * sizeof(unsigned int));
    cudaMemcpy(csr_d->colIdxs, csr->colIdxs, csr->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&csr_d->values , csr->nnz * sizeof(float));
    cudaMemcpy(csr_d->values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice);
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
        
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        spmspm(outBuffer, inBuffer, W_d[layer], bias);
        stopTimeAndPrint(&timer, "");

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

