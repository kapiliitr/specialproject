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
	else
		vertices[v] = -1;
}

__global__ void bfs(const graph_t * graph, int * vertices, int current_depth, bool * d_over)
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
					*d_over = true;
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
		filename = "input.txt";
	}

	FILE * fp = fopen(filename,"r");
	if(!fp)
	{
		printf("Error reading file.\n");
		exit(-1);
	}

	/* Get graph from file into CPU memory  */
	int num_vertices, num_edges, i, j;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	graph_t *graph_host = (graph_t *) malloc(sizeof(graph_t));
	graph_host->V = num_vertices;
	graph_host->adj_prefix_sum = (int *) calloc(num_vertices, sizeof(int));
	graph_host->adj = (int *) malloc(num_edges*sizeof(int *));
	for(i=0; i<num_vertices; i++)
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

	/* Transfer graph data from CPU to GPU memory  */
	graph_t *graph_device;
	int *d_adj_prefix_sum;
	int *d_adj;

	CUDA_SAFE_CALL(cudaMalloc((void **)&graph_device, sizeof(graph_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_adj_prefix_sum, num_vertices*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_adj, num_edges*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(graph_device, graph_host, sizeof(graph_t), cudaMemcpyHostToDevice)); //Copy the graph
	CUDA_SAFE_CALL(cudaMemcpy(d_adj_prefix_sum, graph_host->adj_prefix_sum, num_vertices*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&(graph_device->adj_prefix_sum), &d_adj_prefix_sum, sizeof(int *), cudaMemcpyHostToDevice)); //Copy the poiner to adj
	CUDA_SAFE_CALL(cudaMemcpy(d_adj, graph_host->adj, num_edges*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&(graph_device->adj), &d_adj, sizeof(int *), cudaMemcpyHostToDevice)); //Copy the poiner to adj

	/*****************************************************
	XXX: GPU does not know the size of each adjacency list.
	For that, a new struct containing size of list and list 
	has to be created and passed to GPU memory. Too much hassle.

	OR

	Create 1-D array in the graph itself which contains the 
	size of each list.
	*****************************************************/

	//temp_kernel<<<1,1>>>(graph_device);

	int num_of_blocks = 1;
	int num_of_threads_per_block = num_vertices;

	if(num_vertices>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(num_vertices/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	int * vertices_host;
	int * vertices_device;
	vertices_host = (int *) malloc(num_vertices * sizeof(int));
	CUDA_SAFE_CALL(cudaMalloc((void **)&vertices_device, num_vertices * sizeof(int)));

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);
	
	cudaEvent_t start,end;
	float diff;
	double time = 0;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&end));

	init<<<grid,threads>>> (vertices_device, 0, num_vertices);

	bool stop;
	bool * d_over;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_over, sizeof(bool)));
	
	int k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(cudaEventRecord(start,0));
		bfs<<<grid, threads>>> (graph_device, vertices_device, k, d_over);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaEventRecord(end,0));
		CUDA_SAFE_CALL(cudaEventSynchronize(end));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&diff, start, end));
		time += diff*1.0e-3;

		CUDA_SAFE_CALL(cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost));
		k++;
	}while(stop);

	CUDA_SAFE_CALL(cudaMemcpy((void *)vertices_host, (void *) vertices_device, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

	printf("Number of iterations : %d\n",k);
	for(int i = 0; i < num_vertices; i++)
	{
		printf("Vertex %d Distance %d\n",i,vertices_host[i]);
	}
	printf("Time: %f ms\n",time);

	CUDA_SAFE_CALL(cudaFree(vertices_device));
	CUDA_SAFE_CALL(cudaFree(d_adj));
	CUDA_SAFE_CALL(cudaFree(d_adj_prefix_sum));
	CUDA_SAFE_CALL(cudaFree(graph_device));
	
	free(vertices_host);
	free(graph_host);

	return 0;
}
