/*
 This code has the assumption that the source vertices are sorted in the input file
 Also, the vertices are 0 indexed
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define MAX_THREADS_PER_BLOCK 1024

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

typedef struct __graph
{
	int V;
	int *adj_prefix_sum;
	int *adj;
} graph_t;

__device__ bool d_over;

__global__ void reset()
{
	d_over = false;
}

// Print the graph
__global__ void temp_kernel(graph_t * graph) 
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id == 0)
	{
		int j;
		for(j=0; j<graph->adj_prefix_sum[graph->V-1]; j++)
			printf("%d ",graph->adj[j]);
		printf("\n");
	}
}

__global__ void init(int * vertices, int starting_vertex, int num_vertices)
{
	int v = blockDim.x*blockIdx.x + threadIdx.x;
	if (v==starting_vertex)
		vertices[v] = 0;
	else if(v < num_vertices)
		vertices[v] = -1;
}

__global__ void bfs(const graph_t * graph, int * vertices, int current_depth)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < graph->V)
	{
		if(vertices[id] == current_depth)
		{
			int i;
			if(id == 0) 
				i = 0;
			else
				i = graph->adj_prefix_sum[id-1];
			for(; i < graph->adj_prefix_sum[id]; i++)
			{
				if(vertices[graph->adj[i]] == -1)
				{
					vertices[graph->adj[i]] = current_depth+1;
					d_over = true;
				}
			}
		}
	}
}

int main(int argc, char * argv[])
{
	static char * filename;
	if(argc>2)
	{
		printf("./a.out <filename>\n");
		exit(-1);
	}
	else if(argc==2)
	{
		filename = argv[1];
	}
	else
	{
		filename = "../data/input.txt";
	}

	FILE * fp = fopen(filename,"r");
	if(!fp)
	{
		printf("Error reading file.\n");
		exit(-1);
	}

	/* Set cuda device to K40  */
	CUDA_SAFE_CALL(cudaSetDevice(0));

	/* Get graph from file into CPU memory  */
	int num_vertices, num_edges, i, j;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	graph_t *graph_host;
	CUDA_SAFE_CALL(cudaMallocHost((void **)&graph_host, sizeof(graph_t)));

	graph_host->V = num_vertices;

	CUDA_SAFE_CALL(cudaMallocHost((void **)&(graph_host->adj_prefix_sum), num_vertices*sizeof(int)));

	CUDA_SAFE_CALL(cudaMallocHost((void **)&(graph_host->adj), num_edges*sizeof(int *)));

/*	for(i=0; i<num_vertices; i++)
	{
		int edges_per_vertex;
		fscanf(fp,"%d",&edges_per_vertex);
		if(i>0)
		{
			graph_host->adj_prefix_sum[i] = graph_host->adj_prefix_sum[i-1]+edges_per_vertex;
			j = graph_host->adj_prefix_sum[i-1];
		}
		else
		{
			graph_host->adj_prefix_sum[i] = edges_per_vertex;
			j = 0;
		}
		for(; j<graph_host->adj_prefix_sum[i]; j++)
		{
			fscanf(fp,"%d",&graph_host->adj[j]);
		}
	}
*/

	/*
	   It has been assumed that the source vertices are in sorted order
	 */
	int * temp_adj = (int *) malloc(num_vertices*sizeof(int));
	int s,d,c=0,ps=0,jt;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		if(ps == s)
		{
			temp_adj[c] = d;
			c++;
		}
		else
		{
			//printf("%d %d %d\n",i,ps,s);
			if(ps>0)
			{
				graph_host->adj_prefix_sum[ps] = graph_host->adj_prefix_sum[ps-1]+c;
				j = graph_host->adj_prefix_sum[ps-1];
			}
			else
			{
				graph_host->adj_prefix_sum[ps] = c;
				j = 0;
			}
			jt = j;
			for(; j<graph_host->adj_prefix_sum[ps]; j++)
			{
				graph_host->adj[j] = temp_adj[j-jt];
			}

			temp_adj[0] = d;
			c=1;
			while((++ps)<s)
			{
				graph_host->adj_prefix_sum[ps] = graph_host->adj_prefix_sum[ps-1];
			}
		}
	}
	if(ps>0)
	{
		graph_host->adj_prefix_sum[ps] = graph_host->adj_prefix_sum[ps-1]+c;
		j = graph_host->adj_prefix_sum[ps-1];
	}
	else
	{
		graph_host->adj_prefix_sum[ps] = c;
		j = 0;
	}
	jt = j;
	for(; j<graph_host->adj_prefix_sum[ps]; j++)
	{
		graph_host->adj[j] = temp_adj[j-jt];
	}
	while((++ps)<num_vertices)
	{
		graph_host->adj_prefix_sum[ps] = graph_host->adj_prefix_sum[ps-1];
	}

	/*****************************************************
	XXX: GPU does not know the size of each adjacency list.
	For that, a new struct containing size of list and list 
	has to be created and passed to GPU memory. Too much hassle.

	OR

	Create 1-D array in the graph itself which contains the 
	size of each list.
	*****************************************************/

	//temp_kernel<<<1,1>>>(graph_host);
	
	int num_of_blocks = 1;
	int num_of_threads_per_block = num_vertices;

	if(num_vertices>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(num_vertices/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	int * vertices_host;
	CUDA_SAFE_CALL(cudaMallocHost((void **)&vertices_host, num_vertices*sizeof(int)));

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);
	
	cudaEvent_t start,end;
	float diff;
	double time = 0;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&end));

	init<<<grid,threads>>> (vertices_host, 0, num_vertices);

	bool stop;
	int k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_over, &stop, sizeof(bool),0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		CUDA_SAFE_CALL(cudaEventRecord(start,0));
					
		bfs<<<grid, threads>>> (graph_host, vertices_host, k);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaEventRecord(end,0));
		CUDA_SAFE_CALL(cudaEventSynchronize(end));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&diff, start, end));
		time += diff*1.0e-3;

		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&stop, d_over, sizeof(bool),0, cudaMemcpyDeviceToHost));
		k++;
	}while(stop);

	printf("Number of iterations : %d\n",k);
	for(int i = 0; i < num_vertices; i++)
	{
		printf("Vertex %d Distance %d\n",i,vertices_host[i]);
	}
	printf("Time: %f ms\n",time);

	CUDA_SAFE_CALL(cudaFreeHost(vertices_host));
	CUDA_SAFE_CALL(cudaFreeHost(graph_host->adj));
	CUDA_SAFE_CALL(cudaFreeHost(graph_host->adj_prefix_sum));
	CUDA_SAFE_CALL(cudaFreeHost(graph_host));

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));
	
	return 0;
}
