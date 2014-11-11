#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_EDGES_PER_SHARD 33554432 

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

typedef struct __graph
{
	int E;
	int Vstart;
	int Vend;
	int * adj_prefix_sum;
	int * adj;
} graph_t;

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

/*__global__ void gather_bfs(shard_t * shard, vertex_t * vertices, int current_depth)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < shard->E)
	{
		if(shard->edges[id].val == (current_depth+1))
		{
			int t=shard->edges[id].dest;
			if(vertices[t].val == -1)
			{
				vertices[t].val = current_depth+1;
				d_over = true;
			}
		}
	}
}*/

__global__ void scatter_bfs_vertex(const graph_t * graph, vertex_t * vertices, int current_depth)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int vid = id + graph->Vstart;
	if(vid <= graph->Vend)
	{
		if(vertices[vid].val == current_depth)
		{
			int i;
			if(id == 0) 
				i = 0;
			else
				i = graph->adj_prefix_sum[id-1];
			for(; i < graph->adj_prefix_sum[id]; i++)
			{
				if(vertices[graph->adj[i]].val == -1)
				{
					vertices[graph->adj[i]].val = current_depth+1;
					d_over = true;
				}
			}
		}
	}
}

bool cost(const edge_t &a, const edge_t &b)
{
	return ((a.src < b.src) || (a.src == b.src && a.dest < b.dest));
}

int main(int argc, char * argv[])
{
	struct timeval t1,t2;
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

	printf("Begin file reading...\n");

	/* Get graph from file into CPU memory  */
	int num_vertices, num_edges, i, j, k;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	
	//Array of vectors. vector i contains the in edges of vertex i
	vector< vector<edge_t> > outEdges(num_vertices);
	int * prefixV = (int *) calloc(num_vertices,sizeof(int));
	int s,d,v;

	// In Graphchi case, I am storing the source depth in each edge
	// In X-stream case, I am storing the destination depth in each edge
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		edge_t e;
		e.src=s;
		e.dest=d;
		outEdges[s].push_back(e);
	}
	printf("Finished file reading.\n");
	
	printf("\nBegin interval construction...\n");

	// Construction of intervals
	gettimeofday(&t1,NULL);
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
	gettimeofday(&t2,NULL);
	printf("Time to construct intervals : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));


	printf("\nBegin subgraph construction...\n");
	//Construction of subgraphs
	gettimeofday(&t1,NULL);
	graph_t * subgraph_host = (graph_t *) malloc(num_intervals*sizeof(graph_t));

	//Finding the max number of edges in a shard
	// We will allocate space for that many edges to each shard to maintain consistency
	int MAX_NUM_EDGES_SHARD = INT_MIN;
	int MAX_NUM_VERTICES_SHARD = INT_MIN;

	for(i=0; i<num_intervals; i++)
	{
		int t = prefixV[interval[i].end];
		if(t > MAX_NUM_EDGES_SHARD)
			MAX_NUM_EDGES_SHARD = t;
		int q = interval[i].end-interval[i].start+1;
		if(q > MAX_NUM_VERTICES_SHARD)
			MAX_NUM_VERTICES_SHARD = q;
	}

	for(i=0; i<num_intervals; i++)
	{
		// first and last vertices in shard
		subgraph_host[i].Vstart = interval[i].start;
		subgraph_host[i].Vend = interval[i].end;
		subgraph_host[i].E = prefixV[interval[i].end]; 

		subgraph_host[i].adj_prefix_sum = (int *) malloc(MAX_NUM_VERTICES_SHARD*sizeof(int));
		subgraph_host[i].adj = (int *) malloc(MAX_NUM_EDGES_SHARD*sizeof(int));
	}


	for(i=0; i<num_intervals; i++)
	{
		vector<edge_t> tempEdges;
		for(j=interval[i].start; j<=interval[i].end; j++)
		{
			for(vector<edge_t>::iterator it=outEdges[j].begin(); it!=outEdges[j].end(); ++it)
				tempEdges.push_back(*it);
		}

		//Sorting based on dest vertex to align the edges such that the access of vertices[dest] is sequential
		sort(tempEdges.begin(),tempEdges.end(),cost);

		//TODO: PROBLEM IS IN INTERVAL CONSTRUCTION	
		vector< vector<edge_t> > bucket(MAX_NUM_VERTICES_SHARD);
		for (vector<edge_t>::iterator it = tempEdges.begin() ; it != tempEdges.end(); ++it)
		{
			bucket[(*it).src-interval[i].start].push_back(*it);
		}
		for(j=0;j<MAX_NUM_VERTICES_SHARD;j++)
		{
			subgraph_host[i].adj_prefix_sum[j] = bucket[j].size();
		}
		for(j=1;j<MAX_NUM_VERTICES_SHARD;j++)
		{
			subgraph_host[i].adj_prefix_sum[j] += subgraph_host[i].adj_prefix_sum[j-1];
		}
		k=0;
		for(j=0;j<MAX_NUM_VERTICES_SHARD;j++)
		{
			for (vector<edge_t>::iterator it = bucket[j].begin() ; it != bucket[j].end(); ++it)
			{
				subgraph_host[i].adj[k++] = (*it).dest;
			}
		}


		/*j=0;k=0;
		int p=subgraph_host[i].Vstart;
		for (vector<edge_t>::iterator it = tempEdges.begin() ; it != tempEdges.end(); ++it)
		{
			subgraph_host[i].adj[j] = (*it).dest;
			if(p!=(*it).src)
			{
				while(k<=p)
				{
					subgraph_host[i].adj_prefix_sum[k] = j;
					k++;
				}
				p=(*it).src;
			}
			j++;
		}
		while(k<=p)
		{
			subgraph_host[i].adj_prefix_sum[k] = j;
			k++;
		}*/
	}
	gettimeofday(&t2,NULL);
	printf("Time to construct subgraphs : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));


	int num_of_blocks = 1;
	int num_of_threads_per_block = MAX_NUM_VERTICES_SHARD;

	if(MAX_NUM_VERTICES_SHARD>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(MAX_NUM_VERTICES_SHARD/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	// It will contain the visited status of each vertex
	vertex_t *vertices;
	//CUDA_SAFE_CALL(cudaMallocHost((void **)&vertices, num_vertices*sizeof(vertex_t)));
	vertex_t *vertices_host = (vertex_t *) malloc(num_vertices*sizeof(vertex_t));
	CUDA_SAFE_CALL(cudaMalloc((void **)&vertices, num_vertices*sizeof(vertex_t)));

	init<<<((num_vertices+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK),MAX_THREADS_PER_BLOCK>>> (vertices, 0, num_vertices);

	cudaEvent_t start,end;
	float diff;
	double time = 0;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&end));

	// For vertex centric algo
	graph_t * subgraph_dev;
	int * adj_prefix_sum_dev;
	int * adj_dev;
	CUDA_SAFE_CALL(cudaMalloc((void **)&subgraph_dev, sizeof(graph_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&adj_prefix_sum_dev, MAX_NUM_VERTICES_SHARD*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&adj_dev, MAX_NUM_EDGES_SHARD*sizeof(int)));

	printf("Begin kernel\n");

	bool stop;
	k=0;
	do
	{
		stop = false;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_over, &stop, sizeof(bool),0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		for(i=0; i<num_intervals; i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy(subgraph_dev, &subgraph_host[i], sizeof(graph_t),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(adj_prefix_sum_dev, subgraph_host[i].adj_prefix_sum, MAX_NUM_VERTICES_SHARD*sizeof(int),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(adj_dev, subgraph_host[i].adj, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(&(subgraph_dev->adj_prefix_sum), &adj_prefix_sum_dev, sizeof(int *),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(&(subgraph_dev->adj), &adj_dev, sizeof(int *),cudaMemcpyHostToDevice));


			gettimeofday(&t1,NULL);

			scatter_bfs_vertex<<<grid, threads>>> (subgraph_dev, vertices, k);

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			gettimeofday(&t2,NULL);
			time += ((t2.tv_sec*1.0e3+t2.tv_usec*1.0e-3)-(t1.tv_sec*1.0e3+t1.tv_usec*1.0e-3));
		}
		/*for(i=0; i<num_intervals; i++)
		{
			//Load the data of shard_host[i] into shard (pinned memory)
			shard.E = shard_host[i].E;
			shard.Vstart = shard_host[i].Vstart;
			shard.Vend = shard_host[i].Vend;
			for (j=0; j<shard_host[i].E; j++)
			{
				shard.edges[j] = shard_host[i].edges[j];
				j++;
			}

			gettimeofday(&t1,NULL);

		 	gather_bfs<<<grid, threads>>> (shard, vertices, k, num_vertices);

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			gettimeofday(&t2,NULL);
			time += ((t2.tv_sec*1.0e3+t2.tv_usec*1.0e-3)-(t1.tv_sec*1.0e3+t1.tv_usec*1.0e-3))
		}*/

		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&stop, d_over, sizeof(bool),0, cudaMemcpyDeviceToHost));
		k++;
	}while(stop);

	printf("Number of iterations : %d\n",k);
	CUDA_SAFE_CALL(cudaMemcpy(vertices_host, vertices, num_vertices*sizeof(vertex_t), cudaMemcpyDeviceToHost));
	for(int i = 0; i < num_vertices; i++)
	{
		printf("Vertex %d Distance %d\n",i,vertices_host[i].val);
	}
	printf("Time: %f ms\n",time);

	free(interval);
	for(i=0; i<num_intervals; i++)
	{
		free(subgraph_host[i].adj_prefix_sum);
		free(subgraph_host[i].adj);
	}
	free(subgraph_host);
	free(vertices_host);
	//CUDA_SAFE_CALL(cudaFreeHost(vertices));
	//CUDA_SAFE_CALL(cudaFreeHost(shard->edges));
	//CUDA_SAFE_CALL(cudaFreeHost(shard));
	CUDA_SAFE_CALL(cudaFree(vertices));
	CUDA_SAFE_CALL(cudaFree(adj_prefix_sum_dev));
	CUDA_SAFE_CALL(cudaFree(adj_dev));
	CUDA_SAFE_CALL(cudaFree(subgraph_dev));


	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));

	return 0;
}
