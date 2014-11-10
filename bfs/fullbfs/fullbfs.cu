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

__global__ void scatter_bfs(const shard_t * shard, vertex_t * vertices, int current_depth, int V)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < shard->E)
	{
		int s=shard->edges[id].src;
		if(s < V)
		{
			int t=vertices[s].val;
			if(t==current_depth)
			{
				//shard->edges[id].val = t+1;
				int u=shard->edges[id].dest;
				if(u < V)
				{
					if(vertices[u].val == -1)
					{
						vertices[u].val = t+1;
						d_over = true;
					}
				}
				else
					printf("Illegal vertex dest: %d\n",u);
			}
		}
		else
			printf("Illegal vertex src: %d\n",s);
	}
}


bool cost(const edge_t &a, const edge_t &b)
{
	    return (a.src < b.src);
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


	printf("\nBegin shard construction...\n");
	//Construction of shards
	gettimeofday(&t1,NULL);
	shard_t * shard_host = (shard_t *) malloc(num_intervals*sizeof(shard_t));

	//Finding the max number of edges in a shard
	// We will allocate space for that many edges to each shard to maintain consistency
	int MAX_NUM_EDGES_SHARD = INT_MIN;
	for(i=0; i<num_intervals; i++)
	{
		int t = prefixV[interval[i].end];
		if(t > MAX_NUM_EDGES_SHARD)
			MAX_NUM_EDGES_SHARD = t;
	}

	for(i=0; i<num_intervals; i++)
	{
		// first and last vertices in shard
		shard_host[i].Vstart = interval[i].start;
		shard_host[i].Vend = interval[i].end;

		// number of edges in shard
		shard_host[i].E = prefixV[interval[i].end];
		shard_host[i].edges = (edge_t *) malloc(MAX_NUM_EDGES_SHARD*sizeof(edge_t));
	}


	for(i=0; i<num_intervals; i++)
	{
		vector<edge_t> tempEdges;
		for(j=interval[i].start; j<=interval[i].end; j++)
		{
			for(vector<edge_t>::iterator it=outEdges[j].begin(); it!=outEdges[j].end(); ++it)
				tempEdges.push_back(*it);
		}

		//Sorting based on src vertex to align the edges such that the access of vertices[src] is sequential
		sort(tempEdges.begin(),tempEdges.end(),cost);
		j=0;
		for (vector<edge_t>::iterator it = tempEdges.begin() ; it != tempEdges.end(); ++it)
		{
			shard_host[i].edges[j] = (*it);
			j++;
		}
	}
	gettimeofday(&t2,NULL);
	printf("Time to construct shards : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));

	int num_of_blocks = 1;
	int num_of_threads_per_block = MAX_NUM_EDGES_SHARD;

	if(MAX_NUM_EDGES_SHARD>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(MAX_NUM_EDGES_SHARD/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	shard_t *shard;
	//CUDA_SAFE_CALL(cudaMallocHost((void **)&shard, sizeof(shard_t)));
	//CUDA_SAFE_CALL(cudaMallocHost((void **)&shard->edges, MAX_NUM_EDGES_SHARD*sizeof(edge_t)));
	//CUDA_SAFE_CALL(cudaMallocManaged((void **)&shard, sizeof(shard_t)));
	//CUDA_SAFE_CALL(cudaMallocManaged((void **)&shard->edges, MAX_NUM_EDGES_SHARD*sizeof(edge_t)));

	edge_t * edges_dev;
	CUDA_SAFE_CALL(cudaMalloc((void **)&shard, sizeof(shard_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&edges_dev, MAX_NUM_EDGES_SHARD*sizeof(edge_t)));

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
			//Load the data of shard_host[i] into shard (pinned memory)
			/*shard->E = shard_host[i].E;
			shard->Vstart = shard_host[i].Vstart;
			shard->Vend = shard_host[i].Vend;
			for (j=0; j<shard_host[i].E; j++)
			{
				shard->edges[j] = shard_host[i].edges[j];
				j++;
			}*/
			CUDA_SAFE_CALL(cudaMemcpy(shard, &shard_host[i], sizeof(shard_t),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(edges_dev, shard_host[i].edges, shard_host[i].E*sizeof(edge_t),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(&(shard->edges), &edges_dev, sizeof(edge_t *),cudaMemcpyHostToDevice));

			gettimeofday(&t1,NULL);

			scatter_bfs<<<grid, threads>>> (shard, vertices, k, num_vertices);

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
	/*for(int i = 0; i < num_vertices; i++)
	{
		printf("Vertex %d Distance %d\n",i,vertices_host[i].val);
	}*/
	printf("Time: %f ms\n",time);

	free(interval);
	for(i=0; i<num_intervals; i++)
	{
		free(shard_host[i].edges);
	}
	free(shard_host);
	free(vertices_host);
	//CUDA_SAFE_CALL(cudaFreeHost(vertices));
	//CUDA_SAFE_CALL(cudaFreeHost(shard->edges));
	//CUDA_SAFE_CALL(cudaFreeHost(shard));
	CUDA_SAFE_CALL(cudaFree(vertices));
	CUDA_SAFE_CALL(cudaFree(edges_dev));
	CUDA_SAFE_CALL(cudaFree(shard));


	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(end));

	return 0;
}
