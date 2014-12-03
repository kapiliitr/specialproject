#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <cmath>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define MAX_THREADS_PER_BLOCK 1024
#define GLOBAL_MAX_EDGES_PER_SHARD 33554432 
#define NUM_STREAMS 2
#define ERR 0.01

void safe_call(cudaError_t ret, int line)
{
    if(ret!=cudaSuccess)
    {
        printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
        exit(-1);
    }
}

typedef int VertexId;
typedef int VertexVal;
typedef VertexVal EdgeVal;

typedef struct __interval
{
    VertexId start;
    VertexId end;
} interval_t;

typedef struct __edge
{
    VertexId src;
    VertexId dest;
    EdgeVal val;
} edge_t;

typedef struct __vertex
{
    int numInEdges;
    int numOutEdges;
    VertexVal val;
    //    int shardId;
} vertex_t;

typedef struct __shard
{
    int Ein;
    int Eout;
    VertexId Vstart;
    VertexId Vend;
    VertexId * inEdgesMap;
    VertexId * outEdgesMap;
    edge_t * inEdges;
    edge_t * outEdges;
    VertexVal * inUpdates;
} shard_t;


#ifndef __GATHER__
#define __GATHER__
__global__ void gather(const shard_t * shard, vertex_t * vertices, bool * frontier_cur, bool * frontier_next, int num_vertices, int current_depth)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < shard->Ein)
    {
        if(current_depth==0 && id==0)
        {
            vertices[0].val = 0;
        }
        else
        {
            int d=shard->inEdges[id].dest;
            if(vertices[d].val == -1 && frontier_cur[d] == true)
                vertices[d].val = current_depth;
        }
    }
}

/*
#ifndef __FRONTIER__
#define __FRONTIER__
__global__ void find_frontier(const shard_t * shard, vertex_t * vertices, bool * frontier_cur, bool * frontier_next, int num_vertices, int current_depth)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < shard->Eout)
    {
        int s=shard->outEdges[id].src;
        int d=shard->outEdges[id].dest;
        if(frontier_cur[s] == true && vertices[d].val==-1)
        {
            frontier_next[d] = true;
        }
    }
}
#endif
*/
#endif


#ifndef __APPLY__
#define __APPLY__
__global__ void apply(const shard_t * shard, vertex_t * vertices, bool * frontier_cur, bool * frontier_next, int num_vertices, int current_depth)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    int vid = id + shard->Vstart;
    int d;
    if(vid <= shard->Vend)
    {
        if(frontier_cur[vid] == true)
        {
            int i;
            if(id==0)
                i=0;
            else
                i=shard->outEdgesMap[id-1];
            for(; i<shard->outEdgesMap[id]; i++)
            {
                d = shard->outEdges[i].dest;
                if(vertices[d].val == -1)
                    frontier_next[d] = true;
            }
        }
    }
}
#endif

/*
#ifndef __SCATTER__
#define __SCATTER__
__global__ void scatter(const shard_t * shard, vertex_t * vertices, bool * frontier_cur, bool * frontier_next, int num_vertices, int current_depth)
{
}
#endif
 */

__global__ void reset_frontier(bool * frontier, int V)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < V)
    {
        frontier[id] = false;
    }
}

bool costIn(edge_t a, edge_t b)
{
    return ((a.src < b.src) || (a.src == b.src && a.dest < b.dest));
}

bool costOut(edge_t a, edge_t b)
{
    return ((a.dest < b.dest) || (a.dest == b.dest && a.src < b.src));
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

    //We are always going to have atleast 2 shards to have double bufferring
    int ns = num_edges / GLOBAL_MAX_EDGES_PER_SHARD;
    int MAX_EDGES_PER_SHARD = (ns == 0) ? (num_edges + 1)/2 : (num_edges + 1)/(ns + 1); //We do this to balance the no of edges in the shards

    //Array of vectors. vector i contains the out edges of vertex i
    vector< vector<edge_t> > inEdges(num_vertices);
    vector< vector<edge_t> > outEdges(num_vertices);
    int * prefixVIn = (int *) calloc(num_vertices,sizeof(int));
    int * prefixVOut = (int *) calloc(num_vertices,sizeof(int));
    int s,d;

    // It will contain the visited status of each vertex
    vertex_t *vertices;
    vertex_t *vertices_host = (vertex_t *) malloc(num_vertices*sizeof(vertex_t));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vertices, num_vertices*sizeof(vertex_t)));

    //Initialise the vertices
    for(i=0; i<num_vertices; i++)
    {
        vertices_host[i].numInEdges = 0;
        vertices_host[i].numOutEdges = 0;
        vertices_host[i].val = -1;
    }

    for(i=0; i<num_edges; i++)
    {
        fscanf(fp,"%d",&s);
        fscanf(fp,"%d",&d);
        edge_t e;
        e.src=s;
        e.dest=d;
        inEdges[d].push_back(e);
        outEdges[s].push_back(e);
        vertices_host[s].numOutEdges++;
        vertices_host[d].numInEdges++;
    }
    printf("Finished file reading.\n");

    printf("\nBegin interval construction...\n");

    // Construction of intervals
    gettimeofday(&t1,NULL);
    int num_intervals = 0, add = 1;
    vector<int> startInter;
    prefixVIn[0] = inEdges[0].size();
    prefixVOut[0] = outEdges[0].size();
    if(prefixVIn[0] > MAX_EDGES_PER_SHARD)
    {
        startInter.push_back(0);
        num_intervals++;
        add = 0;
    }
    for(i=1; i<num_vertices; i++)
    {
        prefixVIn[i] = inEdges[i].size();   
        prefixVOut[i] = outEdges[i].size();   
        if(add==1)
        {
            prefixVIn[i] += prefixVIn[i-1];
            prefixVOut[i] += prefixVOut[i-1];
        }
        if(prefixVIn[i] > MAX_EDGES_PER_SHARD)
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
        /*  for(j=interval[i].start; j<=interval[i].end; j++)
            vertices_host[j].shardId = i;*/
    }
    gettimeofday(&t2,NULL);
    printf("Time to construct intervals : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));


    printf("\nBegin shard construction...\n");
    //Construction of shard
    gettimeofday(&t1,NULL);
    shard_t * shard = (shard_t *) malloc(num_intervals*sizeof(shard_t));

    //Finding the max number of edges in a shard
    // We will allocate space for that many edges to each shard to maintain consistency
    int MAX_NUM_INEDGES_SHARD = INT_MIN;
    int MAX_NUM_OUTEDGES_SHARD = INT_MIN;
    int MAX_NUM_VERTICES_SHARD = INT_MIN;

    for(i=0; i<num_intervals; i++)
    {
        int t = prefixVIn[interval[i].end];
        if(t > MAX_NUM_INEDGES_SHARD)
            MAX_NUM_INEDGES_SHARD = t;

        int z = prefixVOut[interval[i].end];
        if(z > MAX_NUM_OUTEDGES_SHARD)
            MAX_NUM_OUTEDGES_SHARD = z;

        int q = interval[i].end-interval[i].start+1;
        if(q > MAX_NUM_VERTICES_SHARD)
            MAX_NUM_VERTICES_SHARD = q;
    }

    for(i=0; i<num_intervals; i++)
    {
        shard[i].Ein = prefixVIn[interval[i].end]; 
        shard[i].Eout = prefixVOut[interval[i].end]; 

        // first and last vertices in shard
        shard[i].Vstart = interval[i].start;
        shard[i].Vend = interval[i].end;

        shard[i].inEdgesMap = (VertexId *) malloc(MAX_NUM_VERTICES_SHARD*sizeof(VertexId));
        shard[i].outEdgesMap = (VertexId *) malloc(MAX_NUM_VERTICES_SHARD*sizeof(VertexId));

        shard[i].inEdges = (edge_t *) malloc(MAX_NUM_INEDGES_SHARD*sizeof(edge_t));
        shard[i].outEdges = (edge_t *) malloc(MAX_NUM_OUTEDGES_SHARD*sizeof(edge_t));

        shard[i].inUpdates = (VertexVal *) malloc(MAX_NUM_INEDGES_SHARD*sizeof(VertexVal));
    }

    for(i=0; i<num_intervals; i++)
    {
        int v = 0, e1 = 0, e2 = 0;
        for(j=interval[i].start; j<=interval[i].end; j++)
        {
            sort(inEdges[j].begin(),inEdges[j].end(),costIn);
            shard[i].inEdgesMap[v] = inEdges[j].size();
            if(v!=0)
                shard[i].inEdgesMap[v] += shard[i].inEdgesMap[v-1];
            for(vector<edge_t>::iterator it=inEdges[j].begin(); it!=inEdges[j].end(); ++it)
            {
                shard[i].inEdges[e1++] = (*it);
            }

            sort(outEdges[j].begin(),outEdges[j].end(),costOut);
            shard[i].outEdgesMap[v] = outEdges[j].size();
            if(v!=0)
                shard[i].outEdgesMap[v] += shard[i].outEdgesMap[v-1];
            for(vector<edge_t>::iterator it=outEdges[j].begin(); it!=outEdges[j].end(); ++it)
            {
                shard[i].outEdges[e2++] = (*it);
            }
            v++;
        }
    }
    gettimeofday(&t2,NULL);
    printf("Time to construct shards : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));

    // It will contain the vertices in the next frontier
    bool *frontier_cur, *frontier_next;
    bool *frontier_host = (bool *) malloc(num_vertices*sizeof(bool));
    frontier_host[0] = true;
    for(i=1; i<num_vertices; i++)
        frontier_host[i] = false;
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_cur, num_vertices*sizeof(bool)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_next, num_vertices*sizeof(bool)));

    /*
       Allocating shards on the device        
     */
    shard_t * shard_dev[NUM_STREAMS];
    VertexId * inEdgesMap_dev[NUM_STREAMS];
    VertexId * outEdgesMap_dev[NUM_STREAMS];
    edge_t * inEdges_dev[NUM_STREAMS];
    edge_t * outEdges_dev[NUM_STREAMS];
    VertexVal * inUpdates_dev[NUM_STREAMS];

    for(int i = 0; i < NUM_STREAMS; i++)
    {
        CUDA_SAFE_CALL(cudaMalloc((void **)&shard_dev[i], sizeof(shard_t)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&inEdgesMap_dev[i], MAX_NUM_VERTICES_SHARD*sizeof(VertexId)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&outEdgesMap_dev[i], MAX_NUM_VERTICES_SHARD*sizeof(VertexId)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&inEdges_dev[i], MAX_NUM_INEDGES_SHARD*sizeof(edge_t)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&outEdges_dev[i], MAX_NUM_OUTEDGES_SHARD*sizeof(edge_t)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&inUpdates_dev[i], MAX_NUM_INEDGES_SHARD*sizeof(VertexVal)));
    }

    // Declaring cuda Streams and events
    cudaStream_t * str;
    cudaEvent_t * start;
    cudaEvent_t * stop;
    str = (cudaStream_t *) malloc(NUM_STREAMS * sizeof(cudaStream_t));
    start = (cudaEvent_t *) malloc(NUM_STREAMS * sizeof(cudaEvent_t));
    stop = (cudaEvent_t *) malloc(NUM_STREAMS * sizeof(cudaEvent_t)); 
    for(int i = 0; i < NUM_STREAMS; i++)
    {
        CUDA_SAFE_CALL(cudaStreamCreate(&(str[i])));
        CUDA_SAFE_CALL(cudaEventCreate(&(start[i])));
        CUDA_SAFE_CALL(cudaEventCreate(&(stop[i])));
    }


    double time = 0;
    float diff;

    /*
       Grid and block dimensions for gather phase (edge centric)
     */
    int num_of_blocks = 1;
    int MAX_THREADS = MAX_NUM_INEDGES_SHARD;
    int num_of_threads_per_block = MAX_THREADS;

    if(MAX_THREADS>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(MAX_THREADS/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    dim3  grid_inedge( num_of_blocks, 1, 1);
    dim3  threads_inedge( num_of_threads_per_block, 1, 1);

    /*
       Grid and block dimensions for apply phase (vertex centric)
     */
    num_of_blocks = 1;
    MAX_THREADS = MAX_NUM_VERTICES_SHARD;
    num_of_threads_per_block = MAX_THREADS;

    if(MAX_THREADS>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(MAX_THREADS/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    dim3  grid_vertex( num_of_blocks, 1, 1);
    dim3  threads_vertex( num_of_threads_per_block, 1, 1);

    /*
       Grid and block dimensions for scatter phase (edge centric)
     */
    num_of_blocks = 1;
    MAX_THREADS = MAX_NUM_OUTEDGES_SHARD;
    num_of_threads_per_block = MAX_THREADS;

    if(MAX_THREADS>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(MAX_THREADS/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    dim3  grid_outedge( num_of_blocks, 1, 1);
    dim3  threads_outedge( num_of_threads_per_block, 1, 1);

    printf("Begin kernel\n");

    CUDA_SAFE_CALL(cudaMemcpy(vertices,vertices_host,num_vertices*sizeof(vertex_t),cudaMemcpyHostToDevice));

    int over, sid;
    k=0;

    gettimeofday(&t1,NULL);

    do
    {
        double tempTime;

        CUDA_SAFE_CALL(cudaMemcpy(frontier_cur,frontier_host,num_vertices*sizeof(bool),cudaMemcpyHostToDevice));
        reset_frontier<<<((num_vertices+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK),MAX_THREADS_PER_BLOCK>>> (frontier_next, num_vertices);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef __GATHER__
        /*
           GATHER PHASE BEGINS
         */
        for(i=0; i<num_intervals; i+=NUM_STREAMS)
        {
            tempTime = 0;
            for(j=0; (j<NUM_STREAMS && (i+j)<num_intervals); j++)
            {
                sid = i+j;
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev[j], &shard[sid], sizeof(shard_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdgesMap_dev[j], shard[sid].inEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdgesMap), &(inEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdgesMap_dev[j], shard[sid].outEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdgesMap), &(outEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdges_dev[j], shard[sid].inEdges, MAX_NUM_INEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdges), &(inEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdges_dev[j], shard[sid].outEdges, MAX_NUM_OUTEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdges), &(outEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inUpdates), &(inUpdates_dev[j]), sizeof(VertexVal *),cudaMemcpyHostToDevice,str[j]));

                CUDA_SAFE_CALL(cudaEventRecord(start[j],str[j]));
                gather<<<grid_inedge, threads_inedge, j, str[j]>>> (shard_dev[j], vertices, frontier_cur, frontier_next, num_vertices, k);
#ifdef __FRONTIER__
                find_frontier<<<grid_outedge, threads_outedge, j, str[j]>>> (shard_dev[j], vertices, frontier_cur, frontier_next, num_vertices, k);
#endif
                CUDA_SAFE_CALL(cudaStreamSynchronize(str[j]));
                CUDA_SAFE_CALL(cudaEventRecord(stop[j],str[j]));
                CUDA_SAFE_CALL(cudaEventSynchronize(stop[j]));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&diff,start[j],stop[j]));
                tempTime += diff;
            }
            time += tempTime;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        /*
           GATHER PHASE ENDS
         */
#endif

#ifdef __APPLY__
        /*
           APPLY PHASE BEGINS
         */
        for(i=0; i<num_intervals; i+=NUM_STREAMS)
        {
            tempTime = 0;
            for(j=0; (j<NUM_STREAMS && (i+j)<num_intervals); j++)
            {
                sid = i+j;
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev[j], &shard[sid], sizeof(shard_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdgesMap_dev[j], shard[sid].inEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdgesMap), &(inEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdgesMap_dev[j], shard[sid].outEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdgesMap), &(outEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdges_dev[j], shard[sid].inEdges, MAX_NUM_INEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdges), &(inEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdges_dev[j], shard[sid].outEdges, MAX_NUM_OUTEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdges), &(outEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inUpdates), &(inUpdates_dev[j]), sizeof(VertexVal *),cudaMemcpyHostToDevice,str[j]));

                CUDA_SAFE_CALL(cudaEventRecord(start[j],str[j]));
                apply<<<grid_vertex, threads_vertex, j, str[j]>>> (shard_dev[j], vertices, frontier_cur, frontier_next, num_vertices, k);
                CUDA_SAFE_CALL(cudaStreamSynchronize(str[j]));
                CUDA_SAFE_CALL(cudaEventRecord(stop[j],str[j]));
                CUDA_SAFE_CALL(cudaEventSynchronize(stop[j]));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&diff,start[j],stop[j]));
                tempTime += diff;
            }
            time += tempTime;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        /*
           APPLY PHASE ENDS
         */
#endif

#ifdef __SCATTER__
        /*
           SCATTER PHASE BEGINS
         */
        for(i=0; i<num_intervals; i+=NUM_STREAMS)
        {
            tempTime = 0;
            for(j=0; (j<NUM_STREAMS && (i+j)<num_intervals); j++)
            {
                sid = i+j;
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev[j], &shard[sid], sizeof(shard_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdgesMap_dev[j], shard[sid].inEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdgesMap), &(inEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdgesMap_dev[j], shard[sid].outEdgesMap, MAX_NUM_VERTICES_SHARD*sizeof(VertexId),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdgesMap), &(outEdgesMap_dev[j]), sizeof(VertexId *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(inEdges_dev[j], shard[sid].inEdges, MAX_NUM_INEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inEdges), &(inEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(outEdges_dev[j], shard[sid].outEdges, MAX_NUM_OUTEDGES_SHARD*sizeof(edge_t),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->outEdges), &(outEdges_dev[j]), sizeof(edge_t *),cudaMemcpyHostToDevice,str[j]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&((shard_dev[j])->inUpdates), &(inUpdates_dev[j]), sizeof(VertexVal *),cudaMemcpyHostToDevice,str[j]));

                CUDA_SAFE_CALL(cudaEventRecord(start[j],str[j]));
                scatter<<<grid_outedge, threads_outedge, j, str[j]>>> (shard_dev[j], vertices, frontier_cur, frontier_next, num_vertices, k);
                CUDA_SAFE_CALL(cudaStreamSynchronize(str[j]));
                CUDA_SAFE_CALL(cudaEventRecord(stop[j],str[j]));
                CUDA_SAFE_CALL(cudaEventSynchronize(stop[j]));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&diff,start[j],stop[j]));
                tempTime += diff;
            }
            time += tempTime;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        /*
           SCATTER PHASE ENDS
         */
#endif

        CUDA_SAFE_CALL(cudaMemcpy(frontier_host,frontier_next,num_vertices*sizeof(bool),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        over=false;
        for(i=0; i<num_vertices; i++)
        {
            if(frontier_host[i])
            {
                over=true;
                break;
            }
        }

        k++;
    }while(over);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    gettimeofday(&t2,NULL);
    printf("Time to BFS : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));

    printf("Number of iterations : %d\n",k);
    /*    CUDA_SAFE_CALL(cudaMemcpy(vertices_host, vertices, num_vertices*sizeof(vertex_t), cudaMemcpyDeviceToHost));
          for(int i = 0; i < num_vertices; i++)
          {
          printf("Vertex %d Distance %d\n",i,(int)vertices_host[i].val);
          }*/

    printf("Time: %f ms\n",time);

    for(int i = 0; i < NUM_STREAMS; i++)
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(str[i]));
        CUDA_SAFE_CALL(cudaEventDestroy(start[i]));
        CUDA_SAFE_CALL(cudaEventDestroy(stop[i]));

        CUDA_SAFE_CALL(cudaFree(inEdgesMap_dev[i]));
        CUDA_SAFE_CALL(cudaFree(outEdgesMap_dev[i]));
        CUDA_SAFE_CALL(cudaFree(inEdges_dev[i]));
        CUDA_SAFE_CALL(cudaFree(outEdges_dev[i]));
        CUDA_SAFE_CALL(cudaFree(inUpdates_dev[i]));
        CUDA_SAFE_CALL(cudaFree(shard_dev[i]));

    }

    free(interval);
    for(i=0; i<num_intervals; i++)
    {
        free(shard[i].inEdgesMap);
        free(shard[i].outEdgesMap);
        free(shard[i].inEdges);
        free(shard[i].outEdges);
        free(shard[i].inUpdates);
    }
    free(shard);
    free(vertices_host);
    free(frontier_host);
    CUDA_SAFE_CALL(cudaFree(vertices));
    CUDA_SAFE_CALL(cudaFree(frontier_cur));
    CUDA_SAFE_CALL(cudaFree(frontier_next));

    return 0;
}
