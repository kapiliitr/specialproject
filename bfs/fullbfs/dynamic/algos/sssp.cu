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
#include <limits.h>

using namespace std;

#define CUDA_SAFE_CALL( err ) (safe_call(err, __LINE__))
#define MAX_THREADS_PER_BLOCK 1024
#define GLOBAL_MAX_EDGES_PER_SHARD 33554432 
#define ERR 1.0E-6

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
    double val;
} edge_t;

typedef struct __vertex
{
    double val;
    int numInEdges;
    int numOutEdges;
} vertex_t;

typedef struct __shard
{
    int E;
    int Vstart;
    int Vend;
    int * from;
    //    int * to;
    double * weight;
    int * vmap;
    double * update;
} shard_t;

__device__ bool d_over;

__global__ void reset()
{
    d_over = false;
}

__device__ int binarySearch(int * arr, int l, int r, int x)
{
    if(x < arr[l])
        return l;
    while (l <= r)
    {
        int m = l+((r-l)/2);
        if (arr[m] == x)
        {
            int t = m;
            while(t<r && arr[t+1] == arr[t])
            {
                t++;
            }
            return t+1;
        }
        if (arr[m] < x) l = m + 1;
        else r = m - 1;
    }
    return l;
}

__global__ void gather_pr_edge(const shard_t * shard, vertex_t * vertices, int num_vertices, int current_depth)
{
    if(current_depth > 0)
    {
        int id = blockDim.x*blockIdx.x + threadIdx.x;
        if(id < shard->E)
        {
            int s = shard->from[id];
	    int V = shard->Vend - shard->Vstart + 1;
            int d = shard->Vstart + binarySearch(shard->vmap, 0, V-1, id);
	    if(vertices[s].val != -1 && vertices[d].val == -1)
	            shard->update[id] = shard->weight[id] + vertices[s].val;
	    else
		    shard->update[id] = -1;
        }
    }
}

__global__ void apply_pr_edge(const shard_t * shard, vertex_t * vertices, int num_vertices, int current_depth)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    int vid = id + shard->Vstart;
    if(vid <= shard->Vend)
    {
        if(current_depth == 0)
        {
            d_over = true;
            if(vid == 0)
                vertices[vid].val = 0;
            else
                vertices[vid].val = -1;
        }
        else
        {
            int i;
            double min=INT_MAX,newval;
            if(id == 0) 
                i = 0;
            else
                i = shard->vmap[id-1];
            for(; i < shard->vmap[id]; i++)
            {
                newval = shard->update[i];
                if(newval != -1)
                {
                    min = (min > newval) ? newval : min;
                }
            }
            if(vertices[vid].val != min && min != INT_MAX)
            {
                d_over=true;
            	vertices[vid].val = min;
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

    //We are always going to have atleast 2 shards to have double bufferring
    int ns = num_edges / GLOBAL_MAX_EDGES_PER_SHARD;
    int MAX_EDGES_PER_SHARD = (ns == 0) ? (num_edges + 1)/2 : (num_edges + 1)/(ns + 1); //We do this to balance the no of edges in the shards

    //Array of vectors. vector i contains the out edges of vertex i
    vector< vector<edge_t> > inEdges(num_vertices);
    int * prefixV = (int *) calloc(num_vertices,sizeof(int));
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
    }

    srand((unsigned) time(0));
    for(i=0; i<num_edges; i++)
    {
        fscanf(fp,"%d",&s);
        fscanf(fp,"%d",&d);
        edge_t e;
        e.src=s;
        e.dest=d;
        e.val=rand()%10+1;
        inEdges[d].push_back(e);
        vertices_host[s].numOutEdges++;
        vertices_host[d].numInEdges++;
    }
    printf("Finished file reading.\n");

    printf("\nBegin interval construction...\n");

    // Construction of intervals
    gettimeofday(&t1,NULL);
    int num_intervals = 0, add = 1;
    vector<int> startInter;
    prefixV[0] = inEdges[0].size();
    if(prefixV[0] > MAX_EDGES_PER_SHARD)
    {
        startInter.push_back(0);
        num_intervals++;
        add = 0;
    }
    for(i=1; i<num_vertices; i++)
    {
        prefixV[i] = inEdges[i].size();    
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
    //Construction of shard
    gettimeofday(&t1,NULL);
    shard_t * shard = (shard_t *) malloc(num_intervals*sizeof(shard_t));

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
        shard[i].Vstart = interval[i].start;
        shard[i].Vend = interval[i].end;
        shard[i].E = prefixV[interval[i].end]; 

        shard[i].vmap = (int *) malloc(MAX_NUM_VERTICES_SHARD*sizeof(int));
        shard[i].from = (int *) malloc(MAX_NUM_EDGES_SHARD*sizeof(int));
        //        shard[i].to = (int *) malloc(MAX_NUM_EDGES_SHARD*sizeof(int));
        shard[i].weight = (double *) malloc(MAX_NUM_EDGES_SHARD*sizeof(double));
        shard[i].update = (double *) malloc(MAX_NUM_EDGES_SHARD*sizeof(double));
    }

    for(i=0; i<num_intervals; i++)
    {
        vector<edge_t> tempEdges;
        for(j=interval[i].start; j<=interval[i].end; j++)
        {
            for(vector<edge_t>::iterator it=inEdges[j].begin(); it!=inEdges[j].end(); ++it)
                tempEdges.push_back(*it);
        }

        //Sorting based on src vertex to align the edges such that the access of vertices[src] is sequential
        sort(tempEdges.begin(),tempEdges.end(),cost);

        vector< vector<edge_t> > bucket(MAX_NUM_VERTICES_SHARD);
        for (vector<edge_t>::iterator it = tempEdges.begin() ; it != tempEdges.end(); ++it)
        {
            bucket[(*it).dest-interval[i].start].push_back(*it);
        }
        for(j=0;j<MAX_NUM_VERTICES_SHARD;j++)
        {
            shard[i].vmap[j] = bucket[j].size();
        }
        for(j=1;j<MAX_NUM_VERTICES_SHARD;j++)
        {
            shard[i].vmap[j] += shard[i].vmap[j-1];
        }
        k=0;
        for(j=0;j<MAX_NUM_VERTICES_SHARD;j++)
        {
            for (vector<edge_t>::iterator it = bucket[j].begin() ; it != bucket[j].end(); ++it)
            {
                shard[i].from[k] = (*it).src;
                //                shard[i].to[k] = (*it).dest;
                shard[i].weight[k] = (*it).val;
                k++;
            }
        }
    }
    gettimeofday(&t2,NULL);
    printf("Time to construct shards : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));

    cudaStream_t * str;
    cudaEvent_t * start;
    cudaEvent_t * stop;
    int num_evts=2;
    str = (cudaStream_t *) malloc(num_evts * sizeof(cudaStream_t));
    start = (cudaEvent_t *) malloc(num_evts * sizeof(cudaEvent_t));
    stop = (cudaEvent_t *) malloc(num_evts * sizeof(cudaEvent_t));
    for(int i = 0; i < num_evts; i++)
    {
        CUDA_SAFE_CALL(cudaStreamCreate(&(str[i])));
        CUDA_SAFE_CALL(cudaEventCreate(&(start[i])));
        CUDA_SAFE_CALL(cudaEventCreate(&(stop[i])));
    }


    float * diff = (float *) malloc(num_intervals*sizeof(float));
    double time = 0;

    // For vertex centric algo
    shard_t * shard_dev;
    int * vmap_dev;
    int * from_dev;
    //    int * to_dev;
    double * weight_dev;
    double * update_dev;
    CUDA_SAFE_CALL(cudaMalloc((void **)&shard_dev, sizeof(shard_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&from_dev, MAX_NUM_EDGES_SHARD*sizeof(int)));
    //    CUDA_SAFE_CALL(cudaMalloc((void **)&to_dev, MAX_NUM_EDGES_SHARD*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&weight_dev, MAX_NUM_EDGES_SHARD*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vmap_dev, MAX_NUM_VERTICES_SHARD*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&update_dev, MAX_NUM_EDGES_SHARD*sizeof(double)));

    //Extra Buffer for doing double bufferring
    shard_t * shard_dev2;
    int * vmap_dev2;
    int * from_dev2;
    //    int * to_dev2;
    double * weight_dev2;
    double * update_dev2;
    CUDA_SAFE_CALL(cudaMalloc((void **)&shard_dev2, sizeof(shard_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&from_dev2, MAX_NUM_EDGES_SHARD*sizeof(int)));
    //    CUDA_SAFE_CALL(cudaMalloc((void **)&to_dev2, MAX_NUM_EDGES_SHARD*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&weight_dev2, MAX_NUM_EDGES_SHARD*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vmap_dev2, MAX_NUM_VERTICES_SHARD*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&update_dev2, MAX_NUM_EDGES_SHARD*sizeof(double)));

    int num_of_blocks = 1;
    //int MAX_THREADS = MAX_NUM_VERTICES_SHARD;
    int MAX_THREADS = MAX_NUM_EDGES_SHARD;
    int num_of_threads_per_block = MAX_THREADS;

    if(MAX_THREADS>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(MAX_THREADS/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    dim3  grid( num_of_blocks, 1, 1);
    dim3  threads( num_of_threads_per_block, 1, 1);

    printf("Begin kernel\n");

    CUDA_SAFE_CALL(cudaMemcpy(vertices,vertices_host,num_vertices*sizeof(vertex_t),cudaMemcpyHostToDevice));

    int pingpong;
    bool over;
    k=0;

    gettimeofday(&t1,NULL);

    do
    {
        over = false;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_over, &over, sizeof(bool),0, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        /*
           GATHER PHASE BEGINS
         */

        pingpong=0;

        for(i=0; i<num_intervals; i++)
        {
            if(pingpong==0)
            {
                //Copy Ping
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev, &shard[i], sizeof(shard_t),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(from_dev, shard[i].from, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->from), &from_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(to_dev, shard[i].to, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->to), &to_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(weight_dev, shard[i].weight, MAX_NUM_EDGES_SHARD*sizeof(double),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->weight), &weight_dev, sizeof(double *),cudaMemcpyHostToDevice,str[0]));

                CUDA_SAFE_CALL(cudaMemcpyAsync(vmap_dev, shard[i].vmap, MAX_NUM_VERTICES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->vmap), &vmap_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->update), &update_dev, sizeof(double *),cudaMemcpyHostToDevice,str[0]));

                if(i>0)
                {
                    //Process Pong
                    CUDA_SAFE_CALL(cudaEventRecord(start[1],str[1]));
                    gather_pr_edge<<<grid, threads,0,str[1]>>> (shard_dev2, vertices, num_vertices, k);
                    CUDA_SAFE_CALL(cudaStreamSynchronize(str[1]));
                    CUDA_SAFE_CALL(cudaEventRecord(stop[1],str[1]));
                    CUDA_SAFE_CALL(cudaEventSynchronize(stop[1]));
                    CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));

                }

                pingpong=1;
            }
            else
            {
                //Copy Pong
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev2, &shard[i], sizeof(shard_t),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(from_dev2, shard[i].from, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->from), &from_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(to_dev2, shard[i].to, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->to), &to_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(weight_dev2, shard[i].weight, MAX_NUM_EDGES_SHARD*sizeof(double),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->weight), &weight_dev2, sizeof(double *),cudaMemcpyHostToDevice,str[1]));

                CUDA_SAFE_CALL(cudaMemcpyAsync(vmap_dev2, shard[i].vmap, MAX_NUM_VERTICES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->vmap), &vmap_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->update), &update_dev2, sizeof(double *),cudaMemcpyHostToDevice,str[0]));

                //Process Pong
                CUDA_SAFE_CALL(cudaEventRecord(start[0],str[0]));
                gather_pr_edge<<<grid, threads,0,str[0]>>> (shard_dev, vertices, num_vertices, k);
                CUDA_SAFE_CALL(cudaStreamSynchronize(str[0]));
                CUDA_SAFE_CALL(cudaEventRecord(stop[0],str[0]));
                CUDA_SAFE_CALL(cudaEventSynchronize(stop[0]));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[0],stop[0]));

                pingpong=0;
            }
        }
        if(pingpong==0)
        {
            //Process Pong
            CUDA_SAFE_CALL(cudaEventRecord(start[1],str[1]));
            gather_pr_edge<<<grid, threads,0,str[1]>>> (shard_dev2, vertices, num_vertices, k);
            CUDA_SAFE_CALL(cudaStreamSynchronize(str[1]));
            CUDA_SAFE_CALL(cudaEventRecord(stop[1],str[1]));
            CUDA_SAFE_CALL(cudaEventSynchronize(stop[1]));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));
        }
        else
        {
            //Process Pong
            CUDA_SAFE_CALL(cudaEventRecord(start[0],str[0]));
            gather_pr_edge<<<grid, threads,0,str[0]>>> (shard_dev, vertices, num_vertices, k);
            CUDA_SAFE_CALL(cudaStreamSynchronize(str[0]));
            CUDA_SAFE_CALL(cudaEventRecord(stop[0],str[0]));
            CUDA_SAFE_CALL(cudaEventSynchronize(stop[0]));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));
        }

        for(i=0;i<num_intervals;i++)
            time += diff[i];

        /*
           GATHER PHASE ENDS
         */

        /*
           APPLY PHASE BEGINS
         */

        pingpong=0;

        for(i=0; i<num_intervals; i++)
        {
            if(pingpong==0)
            {
                //Copy Ping
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev, &shard[i], sizeof(shard_t),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(from_dev, shard[i].from, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->from), &from_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(to_dev, shard[i].to, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->to), &to_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(weight_dev, shard[i].weight, MAX_NUM_EDGES_SHARD*sizeof(double),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->weight), &weight_dev, sizeof(double *),cudaMemcpyHostToDevice,str[0]));

                CUDA_SAFE_CALL(cudaMemcpyAsync(vmap_dev, shard[i].vmap, MAX_NUM_VERTICES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->vmap), &vmap_dev, sizeof(int *),cudaMemcpyHostToDevice,str[0]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev->update), &update_dev, sizeof(double *),cudaMemcpyHostToDevice,str[0]));


                if(i>0)
                {
                    //Process Pong
                    CUDA_SAFE_CALL(cudaEventRecord(start[1],str[1]));
                    apply_pr_edge<<<grid, threads,0,str[1]>>> (shard_dev2, vertices, num_vertices, k);
                    CUDA_SAFE_CALL(cudaStreamSynchronize(str[1]));
                    CUDA_SAFE_CALL(cudaEventRecord(stop[1],str[1]));
                    CUDA_SAFE_CALL(cudaEventSynchronize(stop[1]));
                    CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));

                }

                pingpong=1;
            }
            else
            {
                //Copy Pong
                CUDA_SAFE_CALL(cudaMemcpyAsync(shard_dev2, &shard[i], sizeof(shard_t),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(from_dev2, shard[i].from, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->from), &from_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(to_dev2, shard[i].to, MAX_NUM_EDGES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                //                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->to), &to_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(weight_dev2, shard[i].weight, MAX_NUM_EDGES_SHARD*sizeof(double),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->weight), &weight_dev2, sizeof(double *),cudaMemcpyHostToDevice,str[1]));

                CUDA_SAFE_CALL(cudaMemcpyAsync(vmap_dev2, shard[i].vmap, MAX_NUM_VERTICES_SHARD*sizeof(int),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->vmap), &vmap_dev2, sizeof(int *),cudaMemcpyHostToDevice,str[1]));
                CUDA_SAFE_CALL(cudaMemcpyAsync(&(shard_dev2->update), &update_dev2, sizeof(double *),cudaMemcpyHostToDevice,str[0]));

                //Process Pong
                CUDA_SAFE_CALL(cudaEventRecord(start[0],str[0]));
                apply_pr_edge<<<grid, threads,0,str[0]>>> (shard_dev, vertices, num_vertices, k);
                CUDA_SAFE_CALL(cudaStreamSynchronize(str[0]));
                CUDA_SAFE_CALL(cudaEventRecord(stop[0],str[0]));
                CUDA_SAFE_CALL(cudaEventSynchronize(stop[0]));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[0],stop[0]));

                pingpong=0;
            }
        }
        if(pingpong==0)
        {
            //Process Pong
            CUDA_SAFE_CALL(cudaEventRecord(start[1],str[1]));
            apply_pr_edge<<<grid, threads,0,str[1]>>> (shard_dev2, vertices, num_vertices, k);
            CUDA_SAFE_CALL(cudaStreamSynchronize(str[1]));
            CUDA_SAFE_CALL(cudaEventRecord(stop[1],str[1]));
            CUDA_SAFE_CALL(cudaEventSynchronize(stop[1]));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));
        }
        else
        {
            //Process Pong
            CUDA_SAFE_CALL(cudaEventRecord(start[0],str[0]));
            apply_pr_edge<<<grid, threads,0,str[0]>>> (shard_dev, vertices, num_vertices, k);
            CUDA_SAFE_CALL(cudaStreamSynchronize(str[0]));
            CUDA_SAFE_CALL(cudaEventRecord(stop[0],str[0]));
            CUDA_SAFE_CALL(cudaEventSynchronize(stop[0]));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&diff[i-1],start[1],stop[1]));
        }

        for(i=0;i<num_intervals;i++)
            time += diff[i];

        /*
           APPLY PHASE ENDS
         */

        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&over, d_over, sizeof(bool),0, cudaMemcpyDeviceToHost));
        k++;
    }while(over);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    gettimeofday(&t2,NULL);
    printf("Time to Single Source Shortest Path : %f sec\n",((t2.tv_sec+t2.tv_usec*1.0e-6)-(t1.tv_sec+t1.tv_usec*1.0e-6)));

    printf("Number of iterations : %d\n",k);
    /*    CUDA_SAFE_CALL(cudaMemcpy(vertices_host, vertices, num_vertices*sizeof(vertex_t), cudaMemcpyDeviceToHost));
          for(int i = 0; i < num_vertices; i++)
          {
          printf("Vertex %d Distance %d\n",i,(int)vertices_host[i].val);
          }*/

    printf("Time: %f ms\n",time);

    for(int i = 0; i < num_evts; i++)
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(str[i]));
        CUDA_SAFE_CALL(cudaEventDestroy(start[i]));
        CUDA_SAFE_CALL(cudaEventDestroy(stop[i]));
    }

    free(interval);
    for(i=0; i<num_intervals; i++)
    {
        free(shard[i].vmap);
        free(shard[i].from);
        //        free(shard[i].to);
        free(shard[i].weight);
        free(shard[i].update);
    }
    free(shard);
    free(vertices_host);
    CUDA_SAFE_CALL(cudaFree(vertices));
    CUDA_SAFE_CALL(cudaFree(vmap_dev));
    CUDA_SAFE_CALL(cudaFree(from_dev));
    //    CUDA_SAFE_CALL(cudaFree(to_dev));
    CUDA_SAFE_CALL(cudaFree(weight_dev));
    CUDA_SAFE_CALL(cudaFree(update_dev));
    CUDA_SAFE_CALL(cudaFree(shard_dev));
    CUDA_SAFE_CALL(cudaFree(vmap_dev2));
    CUDA_SAFE_CALL(cudaFree(from_dev2));
    //    CUDA_SAFE_CALL(cudaFree(to_dev2));
    CUDA_SAFE_CALL(cudaFree(weight_dev2));
    CUDA_SAFE_CALL(cudaFree(update_dev2));
    CUDA_SAFE_CALL(cudaFree(shard_dev2));

    return 0;
}
