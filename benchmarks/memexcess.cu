#include <stdio.h>
#include <stdint.h>

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
typedef unsigned long long int LONG;

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

	__global__
void kernel(double * A, LONG N)
{
	LONG i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < N)
		A[i] = (double) i / threadIdx.x;
}

int main()
{
	LONG l;
	scanf("%llu",&l);
	LONG N = (LONG)1 << l;;

	printf("N=%llu Size=%f GB\n",N,N*sizeof(double)/(1024*1024*1024.0));

	const LONG BLOCKSIZE = 1024;
	const LONG NUMBLOCKS = (N + BLOCKSIZE - 1) / BLOCKSIZE;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	/* UVA unpinned memory  */

#if 0

	printf("\nUVA unpinned memory allocation \n");

	double* C_cpu;
	double* D_gpu;

	C_cpu = (double *) malloc(N * sizeof(double));
	printf("Host memory allocated\n");
	CUDA_SAFE_CALL(cudaMalloc((void **)&D_gpu, N * sizeof(double)));
	printf("Device memory allocated\n");

	CUDA_SAFE_CALL(cudaMemcpy((void *)D_gpu, (void *)C_cpu, N * sizeof(double), cudaMemcpyDefault));
	printf("Memory copied to device\n");

	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(D_gpu,N);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy((void *)C_cpu, (void *)D_gpu, N * sizeof(double), cudaMemcpyDefault));
	printf("Memory copied to host\n");

	free(C_cpu);
	CUDA_SAFE_CALL(cudaFree(D_gpu));

#endif

	/* UVA pinned memory  */

#if 0

	printf("\nUVA pinned memory allocation \n");

	double* E_cpu;

	CUDA_SAFE_CALL(cudaHostAlloc ((void **)&E_cpu, N * sizeof(double), cudaHostAllocMapped /*| cudaHostAllocPortable*/));
	printf("Host memory allocated\n");
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(E_cpu,N);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaFreeHost(E_cpu));

#endif

	/* Unified memory  */

#if 1
	
	printf("\nUnified memory allocation \n");
	
	double* F_cpu;

	CUDA_SAFE_CALL(cudaMallocManaged((void **)&F_cpu, N * sizeof(double)));
	printf("Host memory allocated\n");
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(F_cpu,N);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaFree(F_cpu));
#endif

	printf("\nExiting...\n");

	return 0;
}
