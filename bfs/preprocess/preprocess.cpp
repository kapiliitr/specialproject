#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

using namespace std;

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

	FILE * fpo = fopen("output.txt","w");
	if(!fpo)
	{
		printf("Error writing file.\n");
		exit(-1);
	}

	/* Get graph from file into CPU memory  */
	int num_vertices, num_edges, i, j;
	fscanf(fp,"%d %d",&num_vertices,&num_edges);
	set<int> vertices;
	int s,d,max=INT_MIN,u;
	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&d);
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&u);

		vertices.insert(s);
		vertices.insert(d);
		if(s>max)
			max=s;
		if(d>max)
			max=d;
	}
	
	fseek(fp,0,SEEK_SET);
	fscanf(fp,"%d %d",&num_vertices,&num_edges);
	fprintf(fpo,"%d %d\n",num_vertices,num_edges);

	i=0;
	int * vertexIndex = (int *) malloc((max+1)*sizeof(int));
	for(set<int>::iterator it=vertices.begin(); it!=vertices.end(); ++it)
	{
		vertexIndex[(*it)] = i;
		i++;
	}

	for(i=0; i<num_edges; i++)
	{
		fscanf(fp,"%d",&d);
		fscanf(fp,"%d",&s);
		fscanf(fp,"%d",&u);
		fprintf(fpo,"%d ",vertexIndex[d]);
		fprintf(fpo,"%d\n",vertexIndex[s]);
	}

	return 0;
}
