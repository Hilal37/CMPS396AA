
/*
 * Optimization 1: Imnproving thread granularity,
 * by using one thread per output element instead of one per output row.
*/

#include <assert.h>
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define BLOCK_DIM 32

__global__ void spmspm(COOMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {
    
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < A->numRows && col < B->numCols) {
        unsigned int nnzRowA = A->rowPtrs[row + 1] - A->rowPtrs[row];
        unsigned int nnzColB = B->colPtrs[col + 1] - B->colPtrs[col];
        unsigned int ia = A->rowPtrs[row];
        unsigned int ib = B->colPtrs[col];
        unsigned int rowEndA = A->rowPtrs[row + 1];
        unsigned int colEndB = B->colPtrs[col + 1];
        if(nnzRowA > 0 && nnzColB > 0) {
            float sum = 0.0f;
            while(ia < rowEndA && ib < colEndB) {
                unsigned int idxA = A->colIdxs[ia];
                unsigned int idxB = B->rowIdxs[ib];
                if(idxA == idxB) {
                    sum += A->values[ia] * B->values[ib];
			        ia++;
			        ib++;
                }
                else if(idxA < idxB) {
                    ia++;
                }
                else {
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
                    result->rowIdxs[nnzIdx] = row;
                    result->colIdxs[nnzIdx] = col;
                    result->values[nnzIdx] = sum;
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

COOMatrix* createEmptyCOO_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix cooShadow;
    cooShadow.numRows = numRows;
    cooShadow.numCols = numCols;
    cooShadow.nnz = 0;
    cooShadow.capacity = capacity;
    cudaMalloc((void**) &cooShadow.rowIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.values, capacity*sizeof(float));
    COOMatrix* coo_d;
    cudaMalloc((void**) &coo_d, sizeof(COOMatrix));
    cudaMemcpy(coo_d, &cooShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return coo_d;
}

void copyCOOfromGPU(COOMatrix* coo_d, COOMatrix* coo) {
    COOMatrix cooShadow;
    cudaMemcpy(&cooShadow, coo_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    assert(coo->numRows == cooShadow.numRows);
    assert(coo->numCols == cooShadow.numCols);
    assert(coo->capacity >= cooShadow.nnz);
    coo->nnz = cooShadow.nnz;
    cudaMemcpy(coo->rowIdxs, cooShadow.rowIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->colIdxs, cooShadow.colIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->values, cooShadow.values, cooShadow.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

CSRMatrix* createEmptyCSR_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSRMatrix csrShadow;
    csrShadow.numRows = numRows;
    csrShadow.numCols = numCols;
    csrShadow.nnz = 0;
    csrShadow.capacity = capacity;
    cudaMalloc((void**) &csrShadow.rowPtrs, (numRows + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.values, capacity*sizeof(float));
    CSRMatrix* csr_d;
    cudaMalloc((void**) &csr_d, sizeof(CSRMatrix));
    cudaMemcpy(csr_d, &csrShadow, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csr_d;
}

void copyCSRtoGPU(CSRMatrix* csr, CSRMatrix* csr_d) {
    CSRMatrix csrShadow;
    cudaMemcpy(&csrShadow, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    assert(csrShadow.numRows == csr->numRows);
    assert(csrShadow.numCols == csr->numCols);
    assert(csrShadow.capacity >= csr->nnz);
    csrShadow.nnz = csr->nnz;
    cudaMemcpy(csrShadow.rowPtrs, csr->rowPtrs, (csr->numRows + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.colIdxs, csr->colIdxs, csr->nnz*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.values, csr->values, csr->nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

CSCMatrix* createCSCfromCSC_d(CSCMatrix* csc) {
    CSCMatrix cscShadow;
    cscShadow.numRows = csc->numRows;
    cscShadow.numCols = csc->numCols;
    cscShadow.nnz = csc->nnz;
    cscShadow.capacity = csc->capacity;
    cudaMalloc((void**) &cscShadow.colPtrs, (csc->numCols + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.rowIdxs, csc->capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.values, csc->capacity*sizeof(float));
    cudaMemcpy(cscShadow.colPtrs, csc->colPtrs, (csc->numCols + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.rowIdxs, csc->rowIdxs, csc->capacity*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.values, csc->values, csc->capacity*sizeof(float), cudaMemcpyHostToDevice);
    CSCMatrix* csc_d;
    cudaMalloc((void**) &csc_d, sizeof(CSCMatrix));
    cudaMemcpy(csc_d, &cscShadow, sizeof(CSCMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csc_d;
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    Timer timer;

    // Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix* Y0 = createEmptyCSR(featureVectors->numRows, featureVectors->numCols, 4*featureVectors->nnz); // Assuming 4*nnz is enough for all Y vectors
    convertCOOtoCSR(featureVectors, Y0);
    CSRMatrix* Y0_d = createEmptyCSR_d(featureVectors->numRows, featureVectors->numCols, 4*featureVectors->nnz); // Assuming 4*nnz is enough for all Y vectors
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    CSCMatrix* W_d[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCfromCOO(layerWeights[layer]);
        W_d[layer] = createCSCfromCSC_d(W[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");

    // Temporary buffer
    startTime(&timer);
    COOMatrix *tmp = createEmptyCOO(Y0->numRows, Y0->numCols, Y0->capacity);
    COOMatrix *tmp_d = createEmptyCOO_d(Y0->numRows, Y0->numCols, Y0->capacity);
    stopTimeAndPrint(&timer, "Allocate temporary buffer");

    // Loop over layers
    CSRMatrix *Yin = Y0;
    COOMatrix *Yout = tmp;
    CSRMatrix *Yin_d = Y0_d;
    COOMatrix *Yout_d = tmp_d;
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        printf("Computing layer %u (SpMSpM)\n", layer);

        // Copy to GPU
        startTime(&timer);
        copyCSRtoGPU(Yin, Yin_d);
        cudaMemset(&Yout_d->nnz, 0, sizeof(unsigned int));
        stopTimeAndPrint(&timer, "    Copy CSR to GPU and clear COO");

        // SpMSpM
        startTime(&timer);
        // TODO: spmspm <<< ..., ... >>> (Yout_d, Yin_d, W_d[layer], bias);
        dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM);         
	    dim3 numBlocks((Yin->numRows + BLOCK_DIM - 1)/BLOCK_DIM, (W[layer]->numCols + BLOCK_DIM - 1)/BLOCK_DIM);
        spmspm <<< numBlocks, numThreadsPerBlock >>> (Yout_d, Yin_d, W_d[layer], bias);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "    SpMSpM");

        // Copy from GPU
        startTime(&timer);
        copyCOOfromGPU(Yout_d, Yout);
        stopTimeAndPrint(&timer, "    Copy COO from GPU");
        printf("    Output matrix number of nonzeros: %d\n", Yout->nnz);

        // Convert COO to CSR
        startTime(&timer);
        convertCOOtoCSR(Yout, Yin);
        stopTimeAndPrint(&timer, "    Converting COO to CSR");

    }

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, Yin);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSR(Y0);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSC(W[layer]);
    }
    freeCOO(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");

}

