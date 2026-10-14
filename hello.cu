#include<cstdio>

__global__ void hello(){
	printf("Hello from gpu thread %d\n", threadIdx.x);
}

int main(){
	hello<<<1, 8>>>();
	cudaDeviceSynchronize();
	return 0;
}
