#include <iostream>
using namespace std;

__global__ void add(int* data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i]++;
}

int main(){
	const int N = 2048;
	int *a_d, *a_h;
	const int nStreams = 8;

        cudaStream_t stream[nStreams];
	
	for(int i = 0; i < nStreams; i++){
		cudaStreamCreate(&stream[i]);
	}	
	
        const size_t size = N * sizeof(int) / nStreams;
	
	cudaMallocHost(&a_h, N * sizeof(int));
	cudaMalloc(&a_d, N * sizeof(int));
	for(int i = 0; i < N; i++){
	       a_h[i] = i;
	}
	int nThreads = 32;
	for (int i=0; i<nStreams; i++) {
    		int offset = i*N/nStreams;
    		cudaMemcpyAsync(a_d+offset, a_h+offset, size, cudaMemcpyHostToDevice, stream[i]);
    		add<<<N/(nThreads*nStreams), nThreads, 0, stream[i]>>>(a_d+offset);
	}

	cudaDeviceSynchronize();
	cudaMemcpy(a_h, a_d, N * sizeof(int), cudaMemcpyDeviceToHost);
	cout<< a_h[2047];
	cudaFree(a_d);
	cudaFreeHost(a_h);
	return 0;
}
