#include <cstdio>
using namespace std;
global void offsetCopy(float odata, float idata, int offset)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    odata[xid] = idata[xid];
}
int main(){
        float *odata, *idata, *h_data;
        int size = 2048 * sizeof(float);
        cudaMalloc(&odata, size);
        cudaMalloc(&idata, size);
        cudaMallocHost(&h_data, size);
        for(int i = 0; i < 2048; i++){
                h_data[i] = (float)i;
        }
        cudaMemCpy(idata, h_data, size, cudaMemcpyHostToDevice);
        offsetCopy<<<2048 / (32 * 8), 32 * 8>>>(odata, idata, 0);
        cudaDeviceSynchronize();
        cudaFree(odata);
        cudaFree(idata);
        cudaFreeHost(h_data);
        return 0;
}
