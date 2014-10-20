#include <stdio.h>

typedef unsigned long long int LONG;

double bandwidth(LONG n, double t)
{
	return ((double)n * sizeof(double) / t);
}

__global__
void kernel(double * A, LONG N)
{
	LONG i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < N)
		A[i] = (double) i / threadIdx.x;
}

int main(int argc, char *argv[])
{
	LONG N;
	cudaEvent_t start,stop;
	float diff;
	double time, th2d, tunpin, tpin, tmgm;

	if(argc==1)
	{
		N = 100000000;
	}
	else if(argc==2)
	{
		N = atoi(argv[2]);
	}
	else
	{
		printf("./seq <N>");
		exit(-1);
	}
	
	cudaSetDevice(0);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(N=1;N<=1000000000;N=N*10)
	{
	const LONG BLOCKSIZE = 1024;
	const LONG NUMBLOCKS = (N + BLOCKSIZE - 1) / BLOCKSIZE;

	/* Explicit Host to device and vice versa copies  */
	
	double* A_cpu;
	double* B_gpu;

	A_cpu = (double *) malloc(N * sizeof(double));
	cudaMalloc((void **)&B_gpu, N * sizeof(double));
	cudaEventRecord(start, 0);
	cudaMemcpy((void *)B_gpu, (void *)A_cpu, N * sizeof(double), cudaMemcpyHostToDevice);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(B_gpu,N);
	cudaDeviceSynchronize();
	cudaMemcpy((void *)A_cpu, (void *)B_gpu, N * sizeof(double), cudaMemcpyDeviceToHost);
	for(LONG i = 0; i < N; i++)
		A_cpu[i] += i;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;
	//printf("Explicit H2D & D2H bandwidth : %lf GB/s\tTime : %lf s\n",bandwidth(N,time) * 1.0e-9,time);
	th2d = time;

	free(A_cpu);
	cudaFree(B_gpu);

	/* UVA unpinned memory  */
	
	double* C_cpu;
	double* D_gpu;

	C_cpu = (double *) malloc(N * sizeof(double));
	cudaMalloc((void **)&D_gpu, N * sizeof(double));
	cudaEventRecord(start, 0);
	cudaMemcpy((void *)D_gpu, (void *)C_cpu, N * sizeof(double), cudaMemcpyDefault);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(D_gpu,N);
	cudaDeviceSynchronize();
	cudaMemcpy((void *)C_cpu, (void *)D_gpu, N * sizeof(double), cudaMemcpyDefault);

	for(LONG i = 0; i < N; i++)
		C_cpu[i] += i;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);

	time = diff * 1.0e-3;
	//printf("UVA unpinned bandwidth : %lf GB/s\tTime : %lf s\n",bandwidth(N,time) * 1.0e-9,time);
	tunpin = time;

	free(C_cpu);
	cudaFree(D_gpu);

	/* UVA pinned memory  */
	
	double* E_cpu;

	cudaHostAlloc ((void **)&E_cpu, N * sizeof(double), cudaHostAllocMapped /*| cudaHostAllocPortable*/);
	cudaEventRecord(start, 0);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(E_cpu,N);
	cudaDeviceSynchronize();

	for(LONG i = 0; i < N; i++)
		E_cpu[i] += i;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);

	time = diff * 1.0e-3;
	//printf("UVA pinned bandwidth : %lf GB/s\tTime : %lf s\n",bandwidth(N,time) * 1.0e-9,time);
	tpin = time;

	cudaFreeHost(E_cpu);

	/* Unified memory  */
	
#if 1
	double* F_cpu;

	cudaMallocManaged((void **)&F_cpu, N * sizeof(double));
	cudaEventRecord(start, 0);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(F_cpu,N);
	cudaDeviceSynchronize();

	for(LONG i = 0; i < N; i++)
		F_cpu[i] += i;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;
	//printf("Unified memory bandwidth : %lf GB/s\tTime : %lf s\n",bandwidth(N,time) * 1.0e-9,time);
	tmgm = time;
	
	cudaFree(F_cpu);
#endif
	printf("%llu %lf %lf %lf %lf\n",N, th2d, tunpin, tpin, tmgm);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
