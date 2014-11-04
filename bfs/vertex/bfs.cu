#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

using namespace std;

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
	CUDA_SAFE_CALL(cudaMallocManaged((void **)&graph_host, sizeof(graph_t)));

	graph_host->V = num_vertices;

	CUDA_SAFE_CALL(cudaMallocManaged((void **)&(graph_host->adj_prefix_sum), num_vertices*sizeof(int)));

	CUDA_SAFE_CALL(cudaMallocManaged((void **)&(graph_host->adj), num_edges*sizeof(int *)));

	set<int> vertices;
	vector< pair<int,int> > edges;
	int s,d;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		vertices.insert(s);
		vertices.insert(d);
		edges.push_back(make_pair(s,d));
	}
	
	sort(edges.begin(),edges.end());

	/*
	   At this point, the vertices and edges are sorted.
	 */
	i=0;
	int ps=edges[0].first,idf,last;
	for(vector< pair<int,int> >::iterator it = edges.begin() ; it != edges.end(); ++it)
	{
		int id = distance(vertices.begin(),vertices.find((*it).second)); // C++ set stores in sorted order by default
		graph_host->adj[i] = id;
		if(ps != (*it).first)
		{
			idf = distance(vertices.begin(),vertices.find(ps));
			last = distance(vertices.begin(),vertices.find((*it).first));
			for(j=idf;j<last;j++)
				graph_host->adj_prefix_sum[j] = i;
			ps = (*it).first;
		}
		i++;
	}
	idf = distance(vertices.begin(),vertices.find(ps));
	graph_host->adj_prefix_sum[idf] = i;
	for(j=idf;j<num_vertices;j++)
		graph_host->adj_prefix_sum[j] = i;


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
	CUDA_SAFE_CALL(cudaMallocManaged((void **)&vertices_host, num_vertices*sizeof(int)));

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

	CUDA_SAFE_CALL(cudaFree(vertices_host));
	CUDA_SAFE_CALL(cudaFree(graph_host->adj));
	CUDA_SAFE_CALL(cudaFree(graph_host->adj_prefix_sum));
	CUDA_SAFE_CALL(cudaFree(graph_host));

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));
	
	return 0;
}
