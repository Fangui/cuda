#define SIZE     60000000
#define BLOCKDIM 1024

struct Test
{
    Test(unsigned size)
    {
        data = new int[size];
        this->size = size;
    }

    ~Test()
    {
        delete[] data;
    }

    int *data;
    unsigned size;
};

struct Pool
{
    Pool(unsigned size)
    {
        cudaMalloc(&d_test, sizeof(struct Test));
        cudaMalloc(&d_data, size * sizeof(int));
        cudaMemcpy(&(d_test->data), &d_data, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_test->size), &(size), sizeof(unsigned), cudaMemcpyHostToDevice);
    }
    ~Pool()
    {
        cudaFree(d_test);
        cudaFree(d_data);
    }
    Test *d_test;
    int *d_data;
};

__global__ void init(struct Test *t)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t->size)
      return;

    t->data[idx] = idx;
}

void initTest()
{
    Test t(SIZE);

    unsigned size = (SIZE + BLOCKDIM - 1) / BLOCKDIM;

    static Pool p(SIZE);

    init<<<size, BLOCKDIM>>> (p.d_test);

    cudaMemcpy(t.data, p.d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}


int main()
{
    for (unsigned i = 0; i < 10; ++i)
        initTest();

    return 0;
}
