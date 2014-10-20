#include <iostream>
#include <cmath>
#include <cstdio>
#include <sys/time.h>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define BLOCK_SIZE 32
#define ERROR 1.0e-9

typedef unsigned long long int LONG;

void printArr(double *A, LONG N)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
			cout << A[i*N+j] << " ";
		cout << endl;
	}
}

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
	struct timeval t1,t2, tp;
	double tt, gflops;

	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	// Restricted to matrices where N = K*BLOCK_SIZE;
	LONG N,K,S;
	cin >> K >> S;
	N = K*BLOCK_SIZE;
	if(N%S)
	{
		cout << S << " should be divisible by " << N << endl;
		return 0;
	}

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
	double *dA,*dB,*dC,*dAT,*dCT;
	
	// Allocate memory to store the GPU answer on the host
	double *C;
	C = new double[N*N];	

	CUDA_SAFE_CALL(cudaMallocHost(&dB,size));
	CUDA_SAFE_CALL(cudaMallocHost(&dA,(S*size/N)));
	CUDA_SAFE_CALL(cudaMallocHost(&dC,(S*size/N)));
	CUDA_SAFE_CALL(cudaMallocHost(&dAT,(S*size/N)));
	CUDA_SAFE_CALL(cudaMallocHost(&dCT,(S*size/N)));
	
	dim3 threadBlock(BLOCK_SIZE,S);
	dim3 grid(K);

	cudaStream_t * str = (cudaStream_t *) malloc((N/S) * sizeof(cudaStream_t));
	cudaEvent_t * evt = (cudaEvent_t *) malloc((N/S) * sizeof(cudaEvent_t));
	for(int i = 0; i < (N/S); i++)
	{
	        CUDA_SAFE_CALL(cudaStreamCreate(&(str[i])));
		CUDA_SAFE_CALL(cudaEventCreate(&(evt[i])));
	}

	gettimeofday(&t1,0);

	// Copy matrices from the host to device
	CUDA_SAFE_CALL(cudaMemcpyAsync(dB,hB,size,cudaMemcpyHostToDevice,str[0]));

	CUDA_SAFE_CALL(cudaMemcpyAsync(dA,hA,S*(size/N),cudaMemcpyHostToDevice,str[0]));
	gpuMM<<<grid,threadBlock,0,str[0]>>>(dA,dB,dC,N);
	CUDA_SAFE_CALL(cudaEventRecord(evt[0],str[0]));
	for(LONG i=1; i< (N/S); i++){
		if(i%2 == 0)
		{
			//Wait for previous stream to finish executing the kernel
			CUDA_SAFE_CALL(cudaStreamWaitEvent(str[i],evt[i-2],0));

			// Prefetch the next set of rows
			CUDA_SAFE_CALL(cudaMemcpyAsync(dA,hA+i*N*S,(S*size/N),cudaMemcpyHostToDevice,str[i]));

			CUDA_SAFE_CALL(cudaStreamSynchronize(str[i-2]));

			//Execute the matrix multiplication kernel
			gpuMM<<<grid,threadBlock,0,str[i]>>>(dA,dB,dC,N);
			CUDA_SAFE_CALL(cudaEventRecord(evt[i],str[i]));

			// Now copy the GPU result back to CPU
			CUDA_SAFE_CALL(cudaMemcpyAsync(C+(i-1)*N*S,dCT,(S*size/N),cudaMemcpyDeviceToHost,str[i-1]));
		}
		else
		{
			//Wait for previous stream to finish executing the kernel
			if(i>1)
				CUDA_SAFE_CALL(cudaStreamWaitEvent(str[i],evt[i-2],0));

			// Prefetch the next set of rows
			CUDA_SAFE_CALL(cudaMemcpyAsync(dAT,hA+i*N*S,(S*size/N),cudaMemcpyHostToDevice,str[i]));

			if(i>1)
				CUDA_SAFE_CALL(cudaStreamSynchronize(str[i-2]));

			//Execute the matrix multiplication kernel
			gpuMM<<<grid,threadBlock,0,str[i]>>>(dAT,dB,dCT,N);
			CUDA_SAFE_CALL(cudaEventRecord(evt[i],str[i]));

			// Now copy the GPU result back to CPU
			CUDA_SAFE_CALL(cudaMemcpyAsync(C+(i-1)*N*S,dC,(S*size/N),cudaMemcpyDeviceToHost,str[i-1]));
		}
	}
	CUDA_SAFE_CALL(cudaStreamSynchronize(str[(N/S)-1]));
	if(((N/S)-1)%2 == 0)
		CUDA_SAFE_CALL(cudaMemcpyAsync(C+((N/S)-1)*N*S,dC,(S*size/N),cudaMemcpyDeviceToHost,str[(N/S)-1]));
	else
		CUDA_SAFE_CALL(cudaMemcpyAsync(C+((N/S)-1)*N*S,dCT,(S*size/N),cudaMemcpyDeviceToHost,str[(N/S)-1]));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	gettimeofday(&t2,0);
	timersub(&t2,&t1,&tp);

	tt = (double) tp.tv_sec + ((double) tp.tv_usec/1.0e6);
	gflops = ( 1.0e-9 * 2.0 * N * N * N ) / tt; 
	cout << "Prefetch : " << gflops << endl; 

	for(int i = 0; i < (N/S); i++)
	{
		CUDA_SAFE_CALL(cudaStreamDestroy(str[i]));
		CUDA_SAFE_CALL(cudaEventDestroy(evt[i]));
	}

#if 0
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

	// Check the result and make sure it is correct
	for (LONG row=0; row<N; row++){
		for (LONG col=0; col<N; col++){
			if ( fabs(C[row*N+col] - hC[row*N+col]) > ERROR ){
				cout << "Wrong answer!" << row << " " << col << endl;
				row = col = N;
			}
		}
	}
	
	printArr(C,N);
	cout<<endl;
	printArr(hC,N);

#endif

	CUDA_SAFE_CALL(cudaFreeHost(dB));
	CUDA_SAFE_CALL(cudaFreeHost(dA));
	CUDA_SAFE_CALL(cudaFreeHost(dC));
	CUDA_SAFE_CALL(cudaFreeHost(dAT));

	cout << "Finished." << endl;

	return 0;
}
