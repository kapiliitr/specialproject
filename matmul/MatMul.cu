#include <iostream>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define BLOCK_SIZE 32

typedef unsigned long long int LONG;

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		cout << "Error at line " << line << " : " << cudaGetErrorString(ret) << endl;
		exit(-1);
	}
}

__global__ void gpuMM(double *A, double *B, double *C, LONG N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	LONG row = blockIdx.y*blockDim.y + threadIdx.y;
	LONG col = blockIdx.x*blockDim.x + threadIdx.x;

	double sum = 0.f;
	for (LONG n = 0; n < N; ++n)
	    sum += A[row*N+n]*B[n*N+col];

	C[row*N+col] = sum;
}

int main(int argc, char *argv[])
{
	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	// Restricted to matrices where N = K*BLOCK_SIZE;
	LONG N,K;
	cin >> K;
	N = K*BLOCK_SIZE;

	CUDA_SAFE_CALL(cudaSetDevice(1));

	cout << "Executing Matrix Multiplcation" << endl;
	cout << "Matrix size: " << N << "x" << N << endl;

	// Allocate memory on the host
	double *hA,*hB,*hC;
	hA = new double[N*N];
	hB = new double[N*N];
	hC = new double[N*N];

	// Initialize matrices on the host
	for (LONG j=0; j<N; j++){
	    for (LONG i=0; i<N; i++){
	    	hA[j*N+i] = 2.f*(j+i);
			hB[j*N+i] = 1.f*(j-i);
	    }
	}

	// Allocate memory on the device
	LONG size = N*N*sizeof(double);	// Size of the memory in bytes
	double *dA,*dB,*dC;
	CUDA_SAFE_CALL(cudaMalloc(&dA,size));
	CUDA_SAFE_CALL(cudaMalloc(&dB,size));
	CUDA_SAFE_CALL(cudaMalloc(&dC,size));

	cout << "Memory allocated on device memory." << endl;

	dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(K,K);
	
	// Copy matrices from the host to device
	CUDA_SAFE_CALL(cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice));
	
	//Execute the matrix multiplication kernel
	
	gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);
		
	// Now do the matrix multiplication on the CPU
	double sum;
	for (LONG row=0; row<N; row++){
		for (LONG col=0; col<N; col++){
			sum = 0.f;
			for (LONG n=0; n<N; n++){
				sum += hA[row*N+n]*hB[n*N+col];
			}
			hC[row*N+col] = sum;
		}
	}
	
	// Allocate memory to store the GPU answer on the host
	double *C;
	C = new double[N*N];
	
	// Now copy the GPU result back to CPU
	CUDA_SAFE_CALL(cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost));
	
	// Check the result and make sure it is correct
	for (LONG row=0; row<N; row++){
		for (LONG col=0; col<N; col++){
			if ( C[row*N+col] != hC[row*N+col] ){
				cout << "Wrong answer!" << endl;
				row = col = N;
			}
		}
	}
		
	cout << "Finished." << endl;
	
}
