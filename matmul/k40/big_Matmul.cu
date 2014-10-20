#include <iostream>
#include <cmath>
#include <cstdio>
#include <sys/time.h>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define BLOCK_SIZE 32
#define ERROR 1.0e-9

typedef unsigned long long int LONG;

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		cout << "Error at line " << line << " : " << cudaGetErrorString(ret) << endl;
		exit(-1);
	}
}

void printMat(double *A, LONG N)
{
	LONG i,j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			cout << A[i*N+j] << " ";
		cout<<endl;
	}
}

__global__ void gpuMM(double *A, double *B, double *C, LONG N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	LONG row = threadIdx.y;
	LONG col = blockIdx.x*blockDim.x + threadIdx.x;

	double sum = 0.f;
	for (LONG n = 0; n < N; n++)
	    sum += A[row*N+n]*B[n*N+col];

	C[row*N+col] = sum;
}

int main(int argc, char *argv[])
{
	struct timeval t1,t2, tnp;
	double tt, gflops;

	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	// Restricted to matrices where N = K*BLOCK_SIZE;
	LONG N,K;
	cin >> K;
	N = K*BLOCK_SIZE;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	cout << "Executing Matrix Multiplcation" << endl;
	cout << "Matrix size: " << N << "x" << N << endl;

	// Allocate memory on the host
	double *hA,*hB,*hC;
	hA = new double[N*N];
	hB = new double[N*N];
	hC = new double[N*N];

	// Initialize matrices on the host
	srand(time(NULL));
	for (LONG j=0; j<N; j++){
	    for (LONG i=0; i<N; i++){
	    	hA[j*N+i] = drand48();
		hB[j*N+i] = drand48();
	    }
	}

	// Allocate memory on the device
	LONG size = N*N*sizeof(double);	// Size of the memory in bytes
	double *dA,*dB,*dC;
	
	// Allocate memory to store the GPU answer on the host
	double *C;
	C = new double[N*N];	

	CUDA_SAFE_CALL(cudaMalloc(&dB,size));
	CUDA_SAFE_CALL(cudaMalloc(&dA,(K*size/N)));
	CUDA_SAFE_CALL(cudaMalloc(&dC,(K*size/N)));
	
	dim3 threadBlock(BLOCK_SIZE,K);
	dim3 grid(K);

	gettimeofday(&t1,0);
	CUDA_SAFE_CALL(cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice));
	for(LONG i=0; i< (N/K); i++){
		//cout << "Iteration " << i << endl;
	
		CUDA_SAFE_CALL(cudaMemcpy(dA,hA+i*N*K,(K*size/N),cudaMemcpyHostToDevice));
	
		//Execute the matrix multiplication kernel	
		gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);
	
		// Now copy the GPU result back to CPU
		CUDA_SAFE_CALL(cudaMemcpy(C+i*N*K,dC,(K*size/N),cudaMemcpyDeviceToHost));

	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	gettimeofday(&t2,0);
	timersub(&t2,&t1,&tnp);

	tt = (double) tnp.tv_sec + ((double) tnp.tv_usec/1.0e6);
	gflops = ( 1.0e-9 * 2.0 * N * N * N ) / tt; 
	cout << "Without Prefetch : " << gflops << endl; 

	cout << "Device operations done." << endl;

	CUDA_SAFE_CALL(cudaFree(dB));
	CUDA_SAFE_CALL(cudaFree(dA));
	CUDA_SAFE_CALL(cudaFree(dC));

	cout << "Finished." << endl;
	
	return 0;
}
