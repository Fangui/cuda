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

__global__ void
mult_mat(int *mat_a, int *mat_b, int *output)
{
    __shared__ int shared_x[block_size][block_size];
    __shared__ int shared_y[block_size][block_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * block_size + tx;
    int y = blockIdx.y * block_size + ty;
    
    int res = 0;
    for (int i = 0; i < cols / block_size; ++i)
    {
        shared_x[ty][tx] = mat_a[y * cols + i * block_size + tx];
        shared_y[ty][tx] = mat_b[(i * block_size + ty) * lines + x];
        __syncthreads();

        for (int j = 0; j < block_size; ++j)
            res += shared_x[ty][j] * shared_y[j][tx];
        __syncthreads();
    }
    output[y * cols + x] = res;
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

int mat_comp(int *a, int *b)
{
    for (int i = 0; i < lines * cols; ++i)
    {
        if (a[i] != b[i])
            return 1;
    }
    return 0;
}

void mult_ref(int *mat_a, int *mat_b, int *output)
{
    for (unsigned i = 0; i < lines; ++i)
    {
        for (unsigned j = 0; j < lines; ++j)
        {
            output[i * lines + j] = 0;
            for (unsigned k = 0; k < cols; ++k)
            {
               output[i * lines + j] += mat_a[i * cols + k] * mat_b[k * lines + j]; 
            }
        }
    }
}

int main()
{   
    int *mat_a = new int[lines * cols];
    int *mat_b = new int[lines * cols];

    std::size_t nb_bits = lines * cols * sizeof(int);

    dim3 grid(lines / block_size, cols / block_size);
    dim3 block(block_size, block_size);

    #pragma omp simd
    for (int i = 0; i < lines * cols; ++i)
        mat_a[i] = i;

    int *d_in;
    int *d_in_b;
    int *d_out;
    cudaMalloc(&d_in, nb_bits);
    cudaMalloc(&d_out, nb_bits);
    cudaMalloc(&d_in_b, nb_bits);

    cudaMemcpy(d_in, mat_a, nb_bits, cudaMemcpyHostToDevice);

    transpose_matrix <<<grid, block>>> (d_in, d_out);

    cudaMemcpy(mat_b, d_out, nb_bits, cudaMemcpyDeviceToHost); 
      
    //Compare transpose
    int *mat_ref = new int[lines * cols];
    transpose_ref(mat_b, mat_ref);
    cudaMemcpy(mat_b, d_out, nb_bits, cudaMemcpyDeviceToHost); 
    int ret = mat_comp(mat_a, mat_ref);
    if (ret)
        std::cerr << "Error in transpose" << std::endl;

    delete[] mat_ref;
    mat_ref = new int[lines * lines];

    cudaMemcpy(d_in_b, mat_b, lines * lines * sizeof(int), cudaMemcpyHostToDevice);
    mult_mat <<<grid, block>>> (d_in, d_in_b, d_out); 
    //Compare mult mat
    mult_ref(mat_a, mat_b, mat_ref);
    cudaMemcpy(mat_b, d_out, nb_bits, cudaMemcpyDeviceToHost);
    int ret2 = mat_comp(mat_b, mat_ref);
    
    if (ret2)
        std::cerr << "Error in multiplication matrix" << std::endl;

    cudaFree(d_in);
    cudaFree(d_in_b);
    cudaFree(d_out);

    delete[] mat_a;
    delete[] mat_b;
    delete[] mat_ref;

    return ret || ret2;
}
