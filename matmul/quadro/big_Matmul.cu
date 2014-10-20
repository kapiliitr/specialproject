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
	struct timeval t1,t2, tnp, tp, th;
	double tt, gflops;

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
	srand(time(NULL));
	for (LONG j=0; j<N; j++){
	    for (LONG i=0; i<N; i++){
	    	hA[j*N+i] = drand48();
		hB[j*N+i] = drand48();
	    }
	}

	// Allocate memory on the device
	LONG size = N*N*sizeof(double);	// Size of the memory in bytes
	double *dA,*dB,*dC,*dT, *dTemp;
	
	// Allocate memory to store the GPU answer on the host
	double *C;
	C = new double[N*N];	

	CUDA_SAFE_CALL(cudaMalloc(&dB,size));
	CUDA_SAFE_CALL(cudaMalloc(&dA,(K*size/N)));
	CUDA_SAFE_CALL(cudaMalloc(&dC,(K*size/N)));
	CUDA_SAFE_CALL(cudaMalloc(&dT,(K*size/N)));
	
	dim3 threadBlock(BLOCK_SIZE,K);
	dim3 grid(K);

#if 0
	gettimeofday(&t1,0);

	// Copy matrices from the host to device
	CUDA_SAFE_CALL(cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpyAsync(dA,hA,K*(size/N),cudaMemcpyHostToDevice,0));
	gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);
	for(LONG i=1; i< (N/K); i++){
		//cout << "Iteration " << i << endl;

		CUDA_SAFE_CALL(cudaMemcpyAsync(dT,hA+i*N*K,(K*size/N),cudaMemcpyHostToDevice,0));
		CUDA_SAFE_CALL(cudaMemcpy(C+(i-1)*N*K,dC,(K*size/N),cudaMemcpyDeviceToHost));

		//Swap pointers
		dTemp = dT;
		dT = dA;
		dA = dTemp;

		//Execute the matrix multiplication kernel
		gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);

		// Now copy the GPU result back to CPU
		//CUDA_SAFE_CALL(cudaMemcpy(C+i*N,dC,(size/N),cudaMemcpyDeviceToHost));

	}
	CUDA_SAFE_CALL(cudaMemcpy(C+((N/K)-1)*N*K,dC,(K*size/N),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	gettimeofday(&t2,0);
	timersub(&t2,&t1,&tp);
	
	tt = (double) tp.tv_sec + ((double) tp.tv_usec/1.0e6);
	gflops = ( 1.0e-9 * 2.0 * N * N * N ) / tt; 
	cout << "Prefetch : " << gflops << endl; 
#endif

#if 1
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

#endif

	cout << "Device operations done." << endl;

#if 0
	// Now do the matrix multiplication on the CPU
	double sum;
	gettimeofday(&t1,0);
	for (LONG row=0; row<N; row++){
		for (LONG col=0; col<N; col++){
			sum = 0.f;
			for (LONG n=0; n<N; n++){
				sum += hA[row*N+n]*hB[n*N+col];
			}
			hC[row*N+col] = sum;
		}
	}
	gettimeofday(&t2,0);
	timersub(&t2,&t1,&th);
	
	cout << "Host operations done." << endl;

	//printMat(C,N); cout << endl;
	//printMat(hC,N);

	// Check the result and make sure it is correct
	for (LONG row=0; row<N; row++){
		for (LONG col=0; col<N; col++){
			if ( fabs(C[row*N+col] - hC[row*N+col]) > ERROR ){
				cout << "Wrong answer!" << endl;
				row = col = N;
			}
		}
	}
	
	tt = (double) th.tv_sec + ((double) th.tv_usec/1.0e6);
	gflops = ( 1.0e-9 * 2.0 * N * N * N ) / tt; 
	cout << "CPU : " << gflops << endl; 

#endif

	CUDA_SAFE_CALL(cudaFree(dB));
	CUDA_SAFE_CALL(cudaFree(dA));
	CUDA_SAFE_CALL(cudaFree(dC));
	CUDA_SAFE_CALL(cudaFree(dT));

	delete [] hA;
	delete [] hB;
	delete [] hC;
	delete [] C;

	cout << "Finished." << endl;
	
}
