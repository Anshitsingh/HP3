#include <stdio.h>

__global__ void
reduce_kernel_row(int*** g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i =( blockDim.x*blockIdx.x + threadIdx.x)/4;

    if(i < numMats)
    {
        sdata[4*tid] = g_data[0][0][i];
        sdata[4*tid + 1] = g_data[0][1][i] 
        sdata[4*tid + 2] = g_data[1][0][i];
        sdata[4*tid + 3] = g_data[1][1][i];
    }
    else
    {
        sdata[4*tid] = 0;
        sdata[4*tid + 1] = 0;
        sdata[4*tid + 2] = 0;
        sdata[4*tid + 3] = 0;
    }

    __syncthreads();

    for(int s=1;s<blockDim.x;s*=2)
    {
        if(tid%(2*s)==0 && (tid+s)<numMats)
        {
            sdata[4*tid] += sdata[4*(tid + s)];
            sdata[4*tid + 1] += sdata[4*(tid + s) + 1];
            sdata[4*tid + 2] += sdata[4*(tid + s) + 2];
            sdata[4*tid + 3] += sdata[4*(tid + s) + 3];

        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[1];
        g_odata[4*blockIdx.x + 2] = sdata[2];
        g_odata[4*blockIdx.x + 3] = sdata[3];
    }
}

__global__ void
reduce_kernel_row_opt1(int*** g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i =( blockDim.x*blockIdx.x + threadIdx.x)/4;

    if(i < numMats)
    {
        sdata[4*tid] = g_data[0][0][i];
        sdata[4*tid + 1] = g_data[0][1][i] 
        sdata[4*tid + 2] = g_data[1][0][i];
        sdata[4*tid + 3] = g_data[1][1][i];
    }
    else
    {
        sdata[4*tid] = 0;
        sdata[4*tid + 1] = 0;
        sdata[4*tid + 2] = 0;
        sdata[4*tid + 3] = 0;
    }
    __syncthreads();

    for(int s=1;s<blockDim.x;s*=2)
    {
        int index = 2*s*tid;

        if(index<blockDim.x && (index+s)<numMats)
        {
            sdata[4*index] += sdata[4*(index + s)];
            sdata[4*index + 1] += sdata[4*(index + s) + 1];
            sdata[4*index + 2] += sdata[4*(index + s) + 2];
            sdata[4*index + 3] += sdata[4*(index + s) + 3];

        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[1];
        g_odata[4*blockIdx.x + 2] = sdata[2];
        g_odata[4*blockIdx.x + 3] = sdata[3];
    }
}

__global__ void
reduce_kernel_row_opt2(int*** g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i =( blockDim.x*blockIdx.x + threadIdx.x)/4;

    if(i < numMats)
    {
        sdata[4*tid] = g_data[0][0][i];
        sdata[4*tid + 1] = g_data[0][1][i] 
        sdata[4*tid + 2] = g_data[1][0][i];
        sdata[4*tid + 3] = g_data[1][1][i];
    }
    else
    {
        sdata[4*tid] = 0;
        sdata[4*tid + 1] = 0;
        sdata[4*tid + 2] = 0;
        sdata[4*tid + 3] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>0; s/=2)
    {
        if(tid<s)
        {
            sdata[4*tid] += sdata[4*(tid + s)];
            sdata[4*tid + 1] += sdata[4*(tid + s) + 1];
            sdata[4*tid + 2] += sdata[4*(tid + s) + 2];
            sdata[4*tid + 3] += sdata[4*(tid + s) + 3];

        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[1];
        g_odata[4*blockIdx.x + 2] = sdata[2];
        g_odata[4*blockIdx.x + 3] = sdata[3];
    }
}

__global__ void
reduce_kernel_row_opt3(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if(i+blockDim.x < numMats)
    {
        sdata[4*tid] = g_data[0][0][i] + g_data[0][0][i+blockDim.x];
        sdata[4*tid + 1] = g_data[0][1][i] + g_data[0][1][i+blockDim.x];
        sdata[4*tid + 2] = g_data[1][0][i] + g_data[1][0][i+blockDim.x];
        sdata[4*tid + 3] = g_data[1][1][i] + g_data[1][1][i+blockDim.x];
    }
    else if(i < numMats)
    {
        sdata[4*tid] = g_data[0][0][i];
        sdata[4*tid + 1] = g_data[0][1][i];
        sdata[4*tid + 2] =g_data[1][0][i];
        sdata[4*tid + 3] = g_data[1][1][i];
    }
    else
    {
        sdata[4*tid] = 0;
        sdata[4*tid + 1] = 0;
        sdata[4*tid + 2] = 0;
        sdata[4*tid + 3] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>0; s/=2)
    {
        if(tid<s)
        {
            sdata[4*tid] += sdata[4*(tid + s)];
            sdata[4*tid + 1] += sdata[4*(tid + s) + 1];
            sdata[4*tid + 2] += sdata[4*(tid + s) + 2];
            sdata[4*tid + 3] += sdata[4*(tid + s) + 3];
        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[1];
        g_odata[4*blockIdx.x + 2] = sdata[2];
        g_odata[4*blockIdx.x + 3] = sdata[3];
    }
}

__global__ void
reduce_kernel_row_opt4(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if(i+blockDim.x < numMats)
    {
          sdata[4*tid] = g_data[0][0][i] + g_data[0][0][i+blockDim.x];
        sdata[4*tid + 1] = g_data[0][1][i] + g_data[0][1][i+blockDim.x];
        sdata[4*tid + 2] = g_data[1][0][i] + g_data[1][0][i+blockDim.x];
        sdata[4*tid + 3] = g_data[1][1][i] + g_data[1][1][i+blockDim.x];
    }
    else if(i < numMats)
    {
         sdata[4*tid] = g_data[0][0][i];
        sdata[4*tid + 1] = g_data[0][1][i];
        sdata[4*tid + 2] =g_data[1][0][i];
        sdata[4*tid + 3] = g_data[1][1][i];
    }
    else
    {
        sdata[4*tid] = 0;
        sdata[4*tid + 1] = 0;
        sdata[4*tid + 2] = 0;
        sdata[4*tid + 3] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>32; s/=2)
    {
        if(tid<s)
        {
            sdata[4*tid] += sdata[4*(tid + s)];
            sdata[4*tid + 1] += sdata[4*(tid + s) + 1];
            sdata[4*tid + 2] += sdata[4*(tid + s) + 2];
            sdata[4*tid + 3] += sdata[4*(tid + s) + 3];

        }
        __syncthreads();
    }

    if(tid<32 && tid+32<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 32)];
        sdata[4*tid + 1] += sdata[4*(tid + 32) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 32) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 32) + 3];
    }

    if(tid<16 && tid+16<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 16)];
        sdata[4*tid + 1] += sdata[4*(tid + 16) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 16) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 16) + 3];
    }

    if(tid<8 && tid+8<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 8)];
        sdata[4*tid + 1] += sdata[4*(tid + 8) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 8) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 8) + 3];
    }

    if(tid<4 && tid+4<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 4)];
        sdata[4*tid + 1] += sdata[4*(tid + 4) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 4) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 4) + 3];
    }

    if(tid<2 && tid+2<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 2)];
        sdata[4*tid + 1] += sdata[4*(tid + 2) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 2) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 2) + 3];
    }

    if(tid<1 && tid+1<blockDim.x)
    {
        sdata[4*tid] += sdata[4*(tid + 1)];
        sdata[4*tid + 1] += sdata[4*(tid + 1) + 1];
        sdata[4*tid + 2] += sdata[4*(tid + 1) + 2];
        sdata[4*tid + 3] += sdata[4*(tid + 1) + 3];
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[1];
        g_odata[4*blockIdx.x + 2] = sdata[2];
        g_odata[4*blockIdx.x + 3] = sdata[3];
    }
}






__global__ void
reduce_kernel_col(int*** g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<numMats)
    {
        sdata[tid] = g_data[0][0][i];
        sdata[blockDim.x*1 + tid] = g_data[1][0][i];
        sdata[blockDim.x*2 + tid] = g_data[0][1][i];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i];
    }
    else
    {
        sdata[tid] = 0;
        sdata[blockDim.x*1 + tid] = 0;
        sdata[blockDim.x*2 + tid] = 0;
        sdata[blockDim.x*3 + tid] = 0;
    }

    __syncthreads();

    for(int s=1;s<blockDim.x;s*=2)
    {
        if(tid%(2*s)==0)
        {
            if((tid+s) < numMats)
            {
                sdata[tid] += sdata[tid + s];
                sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + s];
                sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + s];
                sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + s];
            }
        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[blockDim.x*1];
        g_odata[4*blockIdx.x + 2] = sdata[blockDim.x*2];
        g_odata[4*blockIdx.x + 3] = sdata[blockDim.x*3];
    }
}

__global__ void
reduce_kernel_col_opt1(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<numMats)
    {
        sdata[tid] = g_data[0][0][i];
        sdata[blockDim.x*1 + tid] = g_data[1][0][i];
        sdata[blockDim.x*2 + tid] = g_data[0][1][i];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i];
    }
    else
    {
        sdata[tid] = 0;
        sdata[blockDim.x*1 + tid] = 0;
        sdata[blockDim.x*2 + tid] = 0;
        sdata[blockDim.x*3 + tid] = 0;
    }

    __syncthreads();

    for(int s=1;s<blockDim.x;s*=2)
    {
        int index = 2*s*tid;

        if(index<blockDim.x && (index+s)<numMats)
        {
            sdata[index] += sdata[index + s];
            sdata[blockDim.x*1 + index] += sdata[blockDim.x*1 + index + s];
            sdata[blockDim.x*2 + index] += sdata[blockDim.x*2 + index + s];
            sdata[blockDim.x*3 + index] += sdata[blockDim.x*3 + index + s];
        }

        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[blockDim.x*1];
        g_odata[4*blockIdx.x + 2] = sdata[blockDim.x*2];
        g_odata[4*blockIdx.x + 3] = sdata[blockDim.x*3];
    }
}

__global__ void
reduce_kernel_col_opt2(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<numMats)
    {
        sdata[tid] = g_data[0][0][i];
        sdata[blockDim.x*1 + tid] = g_data[1][0][i];
        sdata[blockDim.x*2 + tid] = g_data[0][1][i];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i];
    }
    else
    {
        sdata[tid] = 0;
        sdata[blockDim.x*1 + tid] = 0;
        sdata[blockDim.x*2 + tid] = 0;
        sdata[blockDim.x*3 + tid] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>0; s/=2)
    {
        if(tid<s)
        {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + s];
            sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + s];
            sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + s];
        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[blockDim.x*1];
        g_odata[4*blockIdx.x + 2] = sdata[blockDim.x*2];
        g_odata[4*blockIdx.x + 3] = sdata[blockDim.x*3];
    }
}

__global__ void
reduce_kernel_col_opt3(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if(i+blockDim.x < numMats)
    {
        sdata[tid] = g_data[0][0][i] + g_data[0][0][i+blockDim.x];;
        sdata[blockDim.x*1 + tid] = g_data[0][1][i] + g_data[0][1][i+blockDim.x];
        sdata[blockDim.x*2 + tid] =  g_data[1][0][i] + g_data[1][0][i+blockDim.x];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i] + g_data[1][1][i+blockDim.x];
    }
    else if(i < numMats)
    {
        sdata[tid] = g_data[0][0][i] ;
        sdata[blockDim.x*1 + tid] = g_data[0][1][i];
        sdata[blockDim.x*2 + tid] = g_data[1][0][i];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i];
    }
    else
    {
        sdata[tid] = 0;
        sdata[blockDim.x*1 + tid] = 0;
        sdata[blockDim.x*2 + tid] = 0;
        sdata[blockDim.x*3 + tid] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>0; s/=2)
    {
        if(tid<s)
        {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + s];
            sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + s];
            sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + s];

        }
        __syncthreads();
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[blockDim.x*1];
        g_odata[4*blockIdx.x + 2] = sdata[blockDim.x*2];
        g_odata[4*blockIdx.x + 3] = sdata[blockDim.x*3];
    }
}

__global__ void
reduce_kernel_col_opt4(int* g_data, int* g_odata, int numMats)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if(i+blockDim.x < numMats)
    {
        sdata[tid] = g_data[0][0][i] + g_data[0][0][i+blockDim.x];;
        sdata[blockDim.x*1 + tid] = g_data[0][1][i] + g_data[0][1][i+blockDim.x];
        sdata[blockDim.x*2 + tid] =  g_data[1][0][i] + g_data[1][0][i+blockDim.x];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i] + g_data[1][1][i+blockDim.x];
    }
    else if(i < numMats)
    {
        sdata[tid] = g_data[0][0][i] ;
        sdata[blockDim.x*1 + tid] = g_data[0][1][i];
        sdata[blockDim.x*2 + tid] = g_data[1][0][i];
        sdata[blockDim.x*3 + tid] = g_data[1][1][i];
    }
    else
    {
        sdata[tid] = 0;
        sdata[blockDim.x*1 + tid] = 0;
        sdata[blockDim.x*2 + tid] = 0;
        sdata[blockDim.x*3 + tid] = 0;
    }

    __syncthreads();

    for(int s=blockDim.x/2; s>32; s/=2)
    {
        if(tid<s)
        {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + s];
            sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + s];
            sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + s];
        }
        __syncthreads();
    }

    if(tid<32 && tid+32<blockDim.x)
    {
        sdata[tid] += sdata[tid + 32];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 32];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 32];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 32];
    }

    if(tid<16 && tid+16<blockDim.x)
    {
        sdata[tid] += sdata[tid + 16];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 16];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 16];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 16];
    }

    if(tid<8 && tid+8<blockDim.x)
    {
        sdata[tid] += sdata[tid + 8];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 8];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 8];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 8];
    }

    if(tid<4 && tid+4<blockDim.x)
    {
        sdata[tid] += sdata[tid + 4];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 4];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 4];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 4];
    }

    if(tid<2 && tid+2<blockDim.x)
    {
        sdata[tid] += sdata[tid + 2];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 2];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 2];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 2];
    }

    if(tid<1 && tid+1<blockDim.x)
    {
        sdata[tid] += sdata[tid + 1];
        sdata[blockDim.x*1 + tid] += sdata[blockDim.x*1 + tid + 1];
        sdata[blockDim.x*2 + tid] += sdata[blockDim.x*2 + tid + 1];
        sdata[blockDim.x*3 + tid] += sdata[blockDim.x*3 + tid + 1];
    }

    if(tid==0)
    {
        g_odata[4*blockIdx.x] = sdata[0];
        g_odata[4*blockIdx.x + 1] = sdata[blockDim.x*1];
        g_odata[4*blockIdx.x + 2] = sdata[blockDim.x*2];
        g_odata[4*blockIdx.x + 3] = sdata[blockDim.x*3];
    }
}
