## CMPS 396AA Project: Sparse neural networks

### Project topic:
Implementing a sparse neural network in parallel using CUDA.

### Team members:
- Hilal Breiss
- Sohaib El Jundi
- Anis El Rabaa
- Malek Hammad

### Optimizations summary:
- **`gpu0`**: Initial naive implementation
- **`gpu1`**: Improved thread granularity (one thread per output element instead of per row)
- **`gpu2`**: Intra-warp synchronization to increment the non-zeros count (to reduce number of atomic operations)
- **`gpu3`**: Shared memory tiling of the input sparse matrices


