#include<stdio.h>

__global__
void kernel(int * a, unsigned long int n)
{
	unsigned long long int i = blockDim.x*blockIdx.x+threadIdx.x;
	if(i<n)
		a[i] += a[i]*0.5;
}

int main()
{
	unsigned long int N = 2509892096;
	int * A = (int *) malloc(N * sizeof(int));
	int * B;
	cudaMalloc(&B, N * sizeof(int));
	cudaMemcpy(B,A,N * sizeof(int),cudaMemcpyHostToDevice);
	int blocks = (N+1023)/1024;
	kernel<<<blocks,1024>>>(A,N);
	free(A);
	return 0;
}
