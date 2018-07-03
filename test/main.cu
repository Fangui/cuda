#include <iostream>

#define SIZE 60

struct Test
{
    Test()
    {
        data = new int[SIZE];
        size = SIZE;
    }

    ~Test()
    {
        delete[] data;
    }

    int *data;
    unsigned size;
};

__global__ void init(struct Test *t)
{
    t->data[threadIdx.x] = threadIdx.x;
}

void initTest()
{
    Test *t = new Test;

    Test *d_test;
    int *d_tmp;

    cudaMalloc(&d_test, sizeof(struct Test));
    cudaMalloc(&d_tmp, sizeof(int) * SIZE);
    cudaMemcpy(&(d_test->data), &d_tmp, sizeof(int *), cudaMemcpyHostToDevice);

    init<<<1, SIZE>>> (d_test);

    cudaMemcpy(t->data, d_tmp, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_test);
    cudaFree(d_tmp);

    delete t;
}


int main()
{
    for (unsigned i = 0; i < 10; ++i)
        initTest();

    return 0;
}
