#include <iostream>
using namespace std;

__global__ void add(int* data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i]++;
}

int main(){
	const int N = 2048;
	int *a_h, *a_map;
	const size_t size = N * sizeof(int);
	cudaDeviceProp prop;	
	cudaGetDeviceProperties(&prop, 0);//0是GPU编号
	if(!prop.canMapHostMemory){
		cout << "Can not use zero-copy";
		exit(0);
	}
	

	cudaSetDeviceFlags(cudaDeviceMapHost);//启用页锁定内存映射
	cudaHostAlloc(&a_h, size, cudaHostAllocMapped);

	//cudaHostAlloc的分配方式 cudaHostAllocMapped是zero copy的map类型, cudaHostAllocDefault是普通的pinned memory，cudaHostAllocPortable这块 pinned memory 可以被多个 CUDA context/device 更方便地使用
	//cudaMallocHost是cudaHostAlloc的更简单的默认版本
	cudaHostGetDevicePointer(&a_map, a_h, 0);
	
	for(int i = 0; i < N; i++){
	       a_h[i] = i;
	}
	int nThreads = 32;
    		add<<<N/(nThreads), nThreads, 0>>>(a_map);//0是shared memory的大小

	cudaDeviceSynchronize();
	cout<< a_h[2047];
	cudaFreeHost(a_h);//只需要free a_h就好
	return 0;
}
