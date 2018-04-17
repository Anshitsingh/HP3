#include "headers.h"
#include <time.h>
#include <sys/time.h>

double getWallTime()
{
    struct timeval time;
    if(gettimeofday(&time, NULL))
    {
        return 0;
    }
    double wallTime = (double)time.tv_sec + (double)time.tv_usec * 0.000001;
    return wallTime;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n", blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }
}

void getNumBlocksAndThreads1(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n", blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }
}


int main(void)
{
    cudaError_t err = cudaSuccess;
    double start_time, end_time, duration;


    //Entering the number of matrices to be added
    int numMats ;
    printf("Enter thenumber of matrices to be added");
    scanf("%d",&numMats);


    int maxThreads = 1024, maxBlocks = 2147483647; //set using prop.maxGridSize[0] and prop.maxThreadsPerBlock
    size_t dataSize = 4*numMats*sizeof(int);
    int i;

    int inputMats[2][2][numMats];

    for(i=0;i<numMats;i++)
    {
        inputMats[0][0][i] = 1;
        inputMats[0][1][i]= 2;
        inputMats[1][0][i] = 3;
        inputMats[1][1][i] = 4;
    }

    int ***d_input1 = NULL;
    err = cudaMalloc((void ***)&d_input1, dataSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  

    int *d_odata1 = NULL;
    err = cudaMalloc((void **)&d_odata1, dataSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_odata1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_odata2 = NULL;
    err = cudaMalloc((void **)&d_odata2, dataSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_odata2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");

    err = cudaMemcpy(d_input1, inputMats, 2*2*numMats, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int numBlocks = 0, numThreads = 0;

    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    int currElems = numBlocks;

    start_time = getWallTime();

    







    /* Row Major Format */

    // naive version
    reduce_kernel_row<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, numMats);

    err = cudaMemcpy(hTempData, d_odata1, numBlocks*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input1, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_row<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata1, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    




    // optimized version 1
    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);
    currElems = numBlocks;

    start_time = getWallTime();

    err = cudaMemcpy(d_input1, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduce_kernel_row_opt1<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, numMats);

    err = cudaMemcpy(hTempData, d_odata1, numBlocks*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input1, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_row_opt1<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata1, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);








    // optimized version 2
    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);
    currElems = numBlocks;

    start_time = getWallTime();

    err = cudaMemcpy(d_input1, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduce_kernel_row_opt2<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, numMats);

    err = cudaMemcpy(hTempData, d_odata1, numBlocks*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input1, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_row_opt2<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata1, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    // optimized version 3
    getNumBlocksAndThreads1(numMats, maxBlocks, maxThreads, numBlocks, numThreads);
    currElems = numBlocks;

    start_time = getWallTime();

    err = cudaMemcpy(d_input1, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduce_kernel_row_opt3<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, numMats);

    err = cudaMemcpy(hTempData, d_odata1, numBlocks*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    while (currElems > 1)
    {
        getNumBlocksAndThreads1(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input1, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_row_opt3<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, currElems);

        currElems = (currElems + 2*numThreads - 1)/(2*numThreads);

        err = cudaMemcpy(hTempData, d_odata1, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    







    // optimized version 4
    getNumBlocksAndThreads1(numMats, maxBlocks, maxThreads, numBlocks, numThreads);
    currElems = numBlocks;

    start_time = getWallTime();

    err = cudaMemcpy(d_input1, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduce_kernel_row_opt4<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, numMats);

    err = cudaMemcpy(hTempData, d_odata1, numBlocks*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    while (currElems > 1)
    {
        getNumBlocksAndThreads1(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input1, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_row_opt4<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input1, d_odata1, currElems);

        currElems = (currElems + 2*numThreads - 1)/(2*numThreads);

        err = cudaMemcpy(hTempData, d_odata1, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);














    /* Column Major Format */

    // naive version
    err = cudaMemcpy(d_input2, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    start_time = getWallTime();

    reduce_kernel_col<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, numMats);

    err = cudaMemcpy(hTempData, d_odata2, numMats*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    currElems = numBlocks;

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input2, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_col<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata2, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    // optimised version 1
    err = cudaMemcpy(d_input2, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    start_time = getWallTime();

    reduce_kernel_col_opt1<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, numMats);

    err = cudaMemcpy(hTempData, d_odata2, numMats*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    currElems = numBlocks;

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input2, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_col_opt1<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata2, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    // optimised version 2
    err = cudaMemcpy(d_input2, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getNumBlocksAndThreads(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    start_time = getWallTime();

    reduce_kernel_col_opt2<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, numMats);

    err = cudaMemcpy(hTempData, d_odata2, numMats*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    currElems = numBlocks;

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input2, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_col_opt2<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, currElems);

        currElems = (currElems + numThreads - 1)/numThreads;

        err = cudaMemcpy(hTempData, d_odata2, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    // optimised version 3
    err = cudaMemcpy(d_input2, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getNumBlocksAndThreads1(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    start_time = getWallTime();

    reduce_kernel_col_opt3<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, numMats);

    err = cudaMemcpy(hTempData, d_odata2, numMats*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    currElems = numBlocks;

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input2, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_col_opt3<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, currElems);

        currElems = (currElems + 2*numThreads - 1)/(2*numThreads);

        err = cudaMemcpy(hTempData, d_odata2, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);

    // optimised version 4
    err = cudaMemcpy(d_input2, rowMajMatsData, dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rowMajMatsData from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getNumBlocksAndThreads1(numMats, maxBlocks, maxThreads, numBlocks, numThreads);

    start_time = getWallTime();

    reduce_kernel_col_opt4<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, numMats);

    err = cudaMemcpy(hTempData, d_odata2, numMats*4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_input2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    currElems = numBlocks;

    while (currElems > 1)
    {
        getNumBlocksAndThreads(currElems, maxBlocks, maxThreads, numBlocks, numThreads);

        err = cudaMemcpy(d_input2, hTempData, currElems*4*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector hTempData from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        reduce_kernel_col_opt4<<<numBlocks, numThreads, 4*numThreads*sizeof(int)>>>(d_input2, d_odata2, currElems);

        currElems = (currElems + 2*numThreads - 1)/(2*numThreads);

        err = cudaMemcpy(hTempData, d_odata2, currElems*4*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_odata2 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    end_time = getWallTime();
    duration = end_time-start_time;
    printf("%lf\n", duration);


    err = cudaFree(d_input1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_input2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_odata1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_odata1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_odata2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_odata2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(rowMajMatsData);
    free(hTempData);

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
