#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

using namespace std;

typedef struct __graph
{
	int V;
	int *adj_prefix_sum;
	int *adj;
} graph_t;

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

	/* Get graph from file into CPU memory  */
	int num_vertices, num_edges, i, j;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);

	graph_t *graph_host = (graph_t *) malloc(sizeof(graph_t));

	graph_host->V = num_vertices;

	graph_host->adj_prefix_sum = (int *) malloc(num_vertices*sizeof(int));
	graph_host->adj = (int *) malloc(num_edges*sizeof(int));

	set<int> source;
	set<int> dest;
	vector< pair<int,int> > edges;
	int s,d;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		source.insert(s);
		dest.insert(d);
		edges.push_back(make_pair(s,d));
	}

	int sz = source.size();
	int dz = dest.size();

	cout << sz << endl;
	cout << dz << endl;

	int cs=0,cd=0;
	int mins=INT_MAX,mind=INT_MAX,maxs=INT_MIN,maxd=INT_MIN;
	for(set<int>::iterator i=source.begin();i!=source.end();i++)
	{
		int t=*i;
		if(t>=sz)
			cs++;
		if(t<mins)
			mins = t;
		if(t>maxs)
			maxs=t;
	}
	for(set<int>::iterator i=dest.begin();i!=dest.end();i++)
	{
		int t=*i;
		if(t>=dz)
			cd++;
		if(t<mind)
			mind = t;
		if(t>maxd)
			maxd=t;

	}
	cout << mins << " " << maxs << endl; 
	cout << mind << " " << maxd << endl; 
	cout << cs << endl;
	cout << cd << endl;

	sort(edges.begin(),edges.end());
	cout << edges.size() << endl;


#if 0
	cout << __LINE__ << endl;
	vector< vector<int> > adj_list(num_vertices);
	int s,d,jt;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&d);
		if(s>=num_vertices)
			cout << s << " " << d << endl;
		adj_list[s].push_back(d);
	}
	cout << __LINE__ << endl;
	for(i=0; i<num_edges; i++)
	{
		sort(adj_list[i].begin(),adj_list[i].end());
	}
	cout << __LINE__ << endl;

	for(i=0; i<num_vertices; i++)
	{
		int edges_per_vertex = adj_list[i].size();
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
		jt = j;
		for(; j<graph_host->adj_prefix_sum[i]; j++)
		{
			graph_host->adj[j] = adj_list[i][j-jt];
		}
	}
	cout << __LINE__ << endl;
#endif

	return 0;
}
