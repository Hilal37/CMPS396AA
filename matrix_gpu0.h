#ifndef _MATRIX_GPU0_H_
#define _MATRIX_GPU0_H_

typedef struct ELLMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int rowSize;
    unsigned int* colIndices;
    unsigned int* nnzPerRow; //array, number of non-zeros of each row
    float* values;
} ELLMatrix;

ELLMatrix* createEmptyELLMatrix(unsigned int numRows, unsigned int numColumns, unsigned int capacity);
void ELLMatrixAdd(ELLMatrix* matrix, float element, unsigned int row, unsigned int column);
void ELLMatrixExpand(ELLMatrix* matrix, unsigned int newCapacity);
void ELLMatrixFree(ELLMatrix* matrix);


#endif