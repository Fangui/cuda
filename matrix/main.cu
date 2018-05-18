#include <iostream>

const int lines = 1024;
const int cols = 1024;
const int block_size = 16;

__global__ void 
transpose_matrix(int *input, int *output)
{
    int i = blockIdx.x * block_size;
    int j = blockIdx.y * block_size;

    int x = threadIdx.x;
    int y = threadIdx.y;

    __shared__ int block_tr[block_size * block_size];

    block_tr[y * block_size + x] = input[i + x + (j + y) * lines]; 

    __syncthreads();
    output[j + x + (i + y) * lines] = block_tr[x * block_size + y];
}

void transpose_ref(int *input, int *output)
{
    for (int i = 0; i < lines; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[j * lines + i] = input[i * cols + j];
        }
    }
}

int check_transpose(int *a, int *b)
{
    for (int i = 0; i < lines * cols; ++i)
    {
        if (a[i] != b[i])
        {
            std::cerr << "error while transpose" << std::endl;
            return 1;
        }
    }
    return 0;
}

int main()
{   
    int *mat_a = new int[lines * cols];
    int *mat_b = new int[lines * cols];
    std::size_t nb_bits = lines * cols * sizeof(int);

    dim3 blocks(lines / block_size, cols / block_size);
    dim3 thread(block_size, block_size);

    #pragma omp simd
    for (int i = 0; i < lines * cols; ++i)
        mat_a[i] = i;

    int *d_in;
    int *d_out;
    cudaMalloc(&d_in, nb_bits);
    cudaMalloc(&d_out, nb_bits);

    cudaMemcpy(d_in, mat_a, nb_bits, cudaMemcpyHostToDevice);

    transpose_matrix <<<blocks, thread>>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(mat_b, d_out, nb_bits, cudaMemcpyDeviceToHost); 

    cudaFree(d_in);
    cudaFree(d_out);

    int *mat_ref = new int[lines * cols];
    transpose_ref(mat_b, mat_ref);
  
    int ret = check_transpose(mat_a, mat_ref);

    delete[] mat_a;
    delete[] mat_b;
    delete[] mat_ref;

    return ret;
}
