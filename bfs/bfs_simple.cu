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

struct Edge
{
	int first;
	int second;
};

__global__ void init(int * vertices, int starting_vertex, int num_vertices)
{
	int v = blockDim.x*blockIdx.x + threadIdx.x;
	if (v==starting_vertex)
		vertices[v] = 0;
	else
		vertices[v] = -1;
}

__global__ void bfs(const Edge * edges, int * vertices, int current_depth, bool * d_over)
{
	int e = blockDim.x*blockIdx.x + threadIdx.x;
	int vfirst = edges[e].first;
	int dfirst = vertices[vfirst];
	int vsecond = edges[e].second;
	int dsecond = vertices[vsecond];
	if ((dfirst == current_depth) && (dsecond == -1))
	{
		vertices[vsecond] = dfirst + 1;
		*d_over = true;
	}
	if ((dsecond == current_depth) && (dfirst == -1))
	{
		vertices[vfirst] = dsecond + 1;
		*d_over = true;
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

	int num_vertices, num_edges;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	int num_of_blocks = 1;
	int num_of_threads_per_block = num_edges;

	if(num_edges>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(num_edges/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	int * vertices_host;
	Edge * edges_host;
	int * vertices_device;
	Edge * edges_device;

	vertices_host = (int *) malloc(num_vertices * sizeof(int));
	edges_host = (Edge *) malloc(num_edges * sizeof(Edge));
	CUDA_SAFE_CALL(cudaMalloc((void **)&vertices_device, num_vertices * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&edges_device, num_edges * sizeof(Edge)));

	int edge_id = 0;
	for(int i=0;i<num_vertices;i++)
	{
		int edges_per_vertex;
		fscanf(fp,"%d",&edges_per_vertex);
		for(int j=0;j<edges_per_vertex;j++)
		{
			edges_host[edge_id].first = i;
			fscanf(fp,"%d",&edges_host[edge_id].second);
			edge_id++;
		}
	}

	CUDA_SAFE_CALL(cudaMemcpy((void *)edges_device, (void *)edges_host, num_edges * sizeof(Edge), cudaMemcpyHostToDevice));	

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);
	
	init<<<grid,threads>>> (vertices_device, 0, num_vertices);

	bool stop;
	bool * d_over;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_over, sizeof(bool)));
	
	int k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice));
		bfs<<<grid, threads>>> (edges_device, vertices_device, k, d_over);
		CUDA_SAFE_CALL(cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost));
		k++;
	}while(stop);

	CUDA_SAFE_CALL(cudaMemcpy((void *)vertices_host, (void *) vertices_device, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

	printf("Number of iterations : %d\n",k);
	for(int i = 0; i < num_vertices; i++)
	{
		printf("Vertex %d Distance %d\n",i,vertices_host[i]);
	}

	free(vertices_host);
	free(edges_host);

	CUDA_SAFE_CALL(cudaFree(vertices_device));
	CUDA_SAFE_CALL(cudaFree(edges_device));

	return 0;
}
