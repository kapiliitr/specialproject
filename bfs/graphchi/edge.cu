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
#define MAX_EDGES_PER_SHARD 2097152 

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

typedef struct __interval
{
	int start;
	int end;
} interval_t;

typedef struct __edge
{
	int src;
	int dest;
	int val;
} edge_t;

typedef struct __vertex
{
	int val;
} vertex_t;

typedef struct __shard
{
	int E;
	int Vstart;
	int Vend;
	edge_t * edges;
} shard_t;
/*
typedef struct __graph
{
	vertex_t * vertices;
} graph_t;

graph_t * load_subgraph(interval_t, vector<edge_t>);
*/
__device__ bool d_over;

__global__ void reset()
{
	d_over = false;
}

__global__ void init(vertex_t * vertices, int starting_vertex, int num_vertices)
{
	int v = blockDim.x*blockIdx.x + threadIdx.x;
	if (v==starting_vertex)
		vertices[v].val = 0;
	else if(v < num_vertices)
		vertices[v].val = -1;
}

__global__ void gather_bfs(shard_t shard, vertex_t * vertices, int current_depth, int V)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < shard.E)
	{
		if(shard.edges[id].val == (current_depth+1))
		{
			int t=shard.edges[id].dest;
			if(vertices[t].val == -1)
			{
				vertices[t].val = current_depth+1;
				d_over = true;
			}
		}
	}
}

__global__ void scatter_bfs(shard_t shard, vertex_t * vertices, int current_depth, int V)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < shard.E)
	{
		int t=vertices[shard.edges[id].src].val;
		if(t >= 0)
		{
			shard.edges[id].val = t+1;
		}
	}
}


bool cost(const edge_t &a, const edge_t &b)
{
	    return (a.src < b.src);
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
	int num_vertices, num_edges, i, j, k;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	
	//Array of vectors. vector i contains the in edges of vertex i
	vector< vector<edge_t> > outEdges(num_vertices);
	int * prefixV = (int *) calloc(num_vertices,sizeof(int));
	int s,d;

	// In Graphchi case, I am storing the source depth in each edge
	// In X-stream case, I am storing the destination depth in each edge
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		edge_t e;
		e.src=s;
		e.dest=d;
		if(d==0)
			e.val=0;
		else
			e.val=-1;
		outEdges[s].push_back(e);
	}

	// Construction of intervals
	int num_intervals = 0, add = 1;
	vector<int> startInter;
	prefixV[0] = outEdges[0].size();
	if(prefixV[0] > MAX_EDGES_PER_SHARD)
	{
		startInter.push_back(0);
		num_intervals++;
		add = 0;
	}
	for(i=1; i<num_vertices; i++)
	{
		prefixV[i] = outEdges[i].size();	
		if(add==1)
			prefixV[i] += prefixV[i-1];
		if(prefixV[i] > MAX_EDGES_PER_SHARD)
		{
			startInter.push_back(i);
			num_intervals++;
			add = 0;
		}
		else
			add = 1;
	}
	if(add==1)
	{
		startInter.push_back(i-1);
		num_intervals++;
	}


	interval_t * interval = (interval_t *) malloc(num_intervals*sizeof(interval_t));
	for(i=0; i<num_intervals; i++)
	{
		interval[i].start = (i == 0) ? 0 : (startInter[i-1]+1);
		interval[i].end = startInter[i];
	}

	//Construction of shards
	shard_t * shard;
	int MAX_NUM_EDGES_SHARD = INT_MIN;

	CUDA_SAFE_CALL(cudaMallocHost((void **)&shard, num_intervals*sizeof(shard_t)));
	for(i=0; i<num_intervals; i++)
	{
		// first and last vertices in shard
		shard[i].Vstart = interval[i].start;
		shard[i].Vend = interval[i].end;

		// number of edges in shard
		shard[i].E = prefixV[interval[i].end];
		CUDA_SAFE_CALL(cudaMallocHost((void **)&shard[i].edges, shard[i].E*sizeof(edge_t)));

		if(shard[i].E > MAX_NUM_EDGES_SHARD)
			MAX_NUM_EDGES_SHARD = shard[i].E;
	}


	for(i=0; i<num_intervals; i++)
	{
		vector<edge_t> tempEdges;
		for(j=interval[i].start; j<=interval[i].end; j++)
		{
			for(vector<edge_t>::iterator it=outEdges[j].begin(); it!=outEdges[j].end(); ++it)
				tempEdges.push_back(*it);
		}
//		sort(tempEdges.begin(),tempEdges.end(),cost);
		j=0;
		for (vector<edge_t>::iterator it = tempEdges.begin() ; it != tempEdges.end(); ++it)
		{
			shard[i].edges[j] = (*it);
			j++;
		}
	}


	int num_of_blocks = 1;
	int num_of_threads_per_block = MAX_NUM_EDGES_SHARD;

	if(MAX_NUM_EDGES_SHARD>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(MAX_NUM_EDGES_SHARD/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	// It will contain the visited status of each vertex
	vertex_t * vertices;
	CUDA_SAFE_CALL(cudaMallocHost((void **)&vertices, num_vertices*sizeof(vertex_t)));

	init<<<((num_vertices+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK),MAX_THREADS_PER_BLOCK>>> (vertices, 0, num_vertices);

	cudaEvent_t start,end;
	float diff;
	double time = 0;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&end));

	bool stop;
	k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_over, &stop, sizeof(bool),0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		CUDA_SAFE_CALL(cudaEventRecord(start,0));

		for(i=0; i<num_intervals; i++)
			scatter_bfs<<<grid, threads>>> (shard[i], vertices, k, num_vertices);
		for(i=0; i<num_intervals; i++)
		 	gather_bfs<<<grid, threads>>> (shard[i], vertices, k, num_vertices);

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
		printf("Vertex %d Distance %d\n",i,vertices[i].val);
	}
	printf("Time: %f ms\n",time);

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));

	return 0;
}
