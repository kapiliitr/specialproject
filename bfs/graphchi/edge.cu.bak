#include <iostream>
#include <vector>
#include <set>
#include <map>
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

typedef __interval
{
	int start;
	int end;
} interval_t;

typedef struct __edge
{
	int src;
	int dest;
} edge_t;

typedef struct __vertex
{
	int val;
	edge_t * inEdges;
	edge_t * outEdges;
} vertex_t;

typedef __shard
{
	int * intervalEdgeMap;
	edge_t * edges;
} shard_t;

typedef struct __graph
{
	vertex_t * vertices;
} graph_t;

graph_t * load_subgraph(interval_t, vector<edge_t>);

__device__ bool d_over;

__global__ void reset()
{
	d_over = false;
}


int main(int argc, char * argv[])
{
	static char * filename;
	if(argc!=2)
	{
		printf("./a.out <filename>\n");
		exit(-1);
	}
	else
	{
		filename = argv[1];
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

	// Assuming only 1 GB of memory
	int edges_per_vertex= num_edges/num_vertices;
	int num_vertex_interval = (2 << 30) / (edges_per_vertex*sizeof(edge_t));
	int num_interval = num_vertices/num_vertex_interval;

	interval_t * interval = (interval_t *) malloc(num_interval*sizeof(interval_t));
	for(i=0; i<num_interval; i++)
	{
		interval[i].start=i*num_vertex_interval;
		int t=(i+1)*num_vertex_interval-1;
		interval[i].end=(t > num_vertices) ? num_vertices : t;
	}

	shard_t * shard = (shard_t *) malloc(num_interval*shard_t);

	set<int> vertices;
	vector<edge_t> * edges = (vector<edge_t> *) malloc(num_interval*sizeof(vector<edge_t>));
	int s,d;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		vertices.insert(s);
		vertices.insert(d);
	}
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		edge_t e;
		e.src=distance(vertices.begin(),vertices.find(s));
		e.dest=distance(vertices.begin(),vertices.find(d));
		edges[(e.dest/num_vertex_interval)].push_back(e);
	}

	for(i=0; i<num_interval; i++)
	{
		sort(edges[i].begin(),edges[i].end());
		for (vector<edge_t>::iterator it = edges[i].begin() ; it != edges[i].end(); ++it)
		{
		}
	}
	
	cudaEvent_t start,end;
	float diff;
	double time = 0;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&end));

	bool stop;
	int k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_over, &stop, sizeof(bool),0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		CUDA_SAFE_CALL(cudaEventRecord(start,0));
	
		for(i = 0; i<P; i++)
		{
			graph_t * subgraph = load_subgraph(interval[i]);
			bfs<<<grid, threads>>> (graph_host, k);
		}

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

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));

	return 0;
}

graph_t * load_subgraph(interval_t interval, vector<edge_t> edges)
{
	int a = interval.start;
	int b = interval.end;

	graph_t * G = (graph_t *) malloc(sizeof(graph_t));
	vertex_t * vertices = (vertex_t *) malloc((b-a+1)*sizeof(vertex_t));
	G->vertices = vertices;
	


	return G;
}
