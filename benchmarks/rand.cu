#include <stdio.h>

typedef unsigned long long int LONG;

void initarr(int *arr, LONG n)
{
	for(LONG i=0; i<n; i++)
		arr[i] = i;
}

void shuffle(int *arr, LONG n)
{
	initarr(arr,n);
	if (n > 1) 
	{
		LONG i;
		srand(time(NULL));
		for (i = 0; i < n - 1; i++) 
		{
			LONG j = i + rand() / (RAND_MAX / (n - i) + 1);
			LONG t = arr[j];
			arr[j] = arr[i];
			arr[i] = t;
		}
	}
}

double bandwidth(LONG n, double t)
{
	return ((double)n * (sizeof(double) + sizeof(int)) / t);
}

__global__
void kernel(double * A, int * T, LONG N)
{
	LONG i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < N)
		A[T[i]] = (double) i / threadIdx.x;
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

	int ka;
	for(ka=1;ka<=4096;ka++)
	{
	N=512*ka;
	const LONG BLOCKSIZE = 1024;
	const LONG NUMBLOCKS = (N + BLOCKSIZE - 1) / BLOCKSIZE;

	/* Explicit Host to device and vice versa copies  */
	
	double* A_cpu; int* T_cpu;
	double* B_gpu; int* T_gpu;

	A_cpu = (double *) malloc(N * sizeof(double));
	cudaMalloc((void **)&B_gpu, N * sizeof(double));

	T_cpu = (int *) malloc(N * sizeof(int));
	shuffle(T_cpu, N);
	cudaMalloc((void **)&T_gpu, N * sizeof(int));
	
	cudaEventRecord(start, 0);
	cudaMemcpy((void *)B_gpu, (void *)A_cpu, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)T_gpu, (void *)T_cpu, N * sizeof(int), cudaMemcpyHostToDevice);

	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(B_gpu,T_gpu,N);
	cudaDeviceSynchronize();	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;

	cudaMemcpy((void *)A_cpu, (void *)B_gpu, N * sizeof(double), cudaMemcpyDeviceToHost);
	for(LONG i = 0; i < N; i++)
		A_cpu[T_cpu[i]] += i;


	//printf("Explicit H2D & D2H bandwidth : %lf GB/s\n",bandwidth(N,time) * 1.0e-9);
	th2d = time;

	free(A_cpu); free(T_cpu);
	cudaFree(B_gpu); cudaFree(T_gpu);

	/* UVA unpinned memory  */
	
	double* C_cpu;
	double* D_gpu;

	C_cpu = (double *) malloc(N * sizeof(double));
	cudaMalloc((void **)&D_gpu, N * sizeof(double));

	T_cpu = (int *) malloc(N * sizeof(int));
	shuffle(T_cpu, N);
	cudaMalloc((void **)&T_gpu, N * sizeof(int));

	cudaEventRecord(start, 0);
	cudaMemcpy((void *)D_gpu, (void *)C_cpu, N * sizeof(double), cudaMemcpyDefault);
	cudaMemcpy((void *)T_gpu, (void *)T_cpu, N * sizeof(int), cudaMemcpyDefault);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(D_gpu,T_gpu,N);
	cudaDeviceSynchronize();	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;

	cudaMemcpy((void *)C_cpu, (void *)D_gpu, N * sizeof(double), cudaMemcpyDefault);
	for(LONG i = 0; i < N; i++)
		C_cpu[T_cpu[i]] += i;


	//printf("UVA unpinned bandwidth : %lf GB/s\n",bandwidth(N,time) * 1.0e-9);
	tunpin = time;

	free(C_cpu);
	cudaFree(D_gpu);

	/* UVA pinned memory  */
	
	double* E_cpu;

	cudaHostAlloc ((void **)&E_cpu, N * sizeof(double), cudaHostAllocMapped /*| cudaHostAllocPortable*/);

	cudaHostAlloc ((void **)&T_cpu, N * sizeof(int), cudaHostAllocMapped /*| cudaHostAllocPortable*/);
	shuffle(T_cpu, N);
	
	cudaEventRecord(start, 0);
	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(E_cpu,T_cpu,N);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;

	for(LONG i = 0; i < N; i++)
		E_cpu[T_cpu[i]] += i;

	
	//printf("UVA pinned bandwidth : %lf GB/s\n",bandwidth(N,time) * 1.0e-9);
	tpin = time;

	cudaFreeHost(E_cpu);
	cudaFreeHost(T_cpu);

	/* Unified memory  */
	
	double* F_cpu;

	cudaMallocManaged((void **)&F_cpu, N * sizeof(double));

	cudaMallocManaged((void **)&T_cpu, N * sizeof(int));
	shuffle(T_cpu, N);
	
	cudaEventRecord(start, 0);

	kernel<<<NUMBLOCKS, BLOCKSIZE>>>(F_cpu,T_cpu,N);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diff,start,stop);
	time = diff * 1.0e-3;

	for(LONG i = 0; i < N; i++)
		F_cpu[T_cpu[i]] += i;


	//printf("Unified memory bandwidth : %lf GB/s\n",bandwidth(N,time) * 1.0e-9);
	tmgm = time;
	
	cudaFree(F_cpu);
	cudaFree(T_cpu);

	printf("%d %lf %lf %lf %lf\n",ka, th2d, tunpin, tpin, tmgm);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
