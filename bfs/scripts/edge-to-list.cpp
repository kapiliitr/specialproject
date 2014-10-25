// Requires C++ 11 support
// Compile with -std=c++0x

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cstring>
#include <set>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iomanip>

using namespace std;

int main()
{
	char file[200] = "../data/facebook_combined.txt";
	set<string> vertices;
	ifstream edgelist(file,ifstream::in);
	string line;
	
	if(!edgelist.good())
	{
		cerr << "Error opening graph file." << endl;
		exit(-1);
	}

	vector<string> from;
	vector<string> to;
	while(getline(edgelist,line))
	{
		vector<string> splitvec;
		char * cstr = new char [line.length()+1];
		strcpy(cstr,line.c_str());
		char * pch;
		pch = strtok (cstr," \t");
		while (pch != NULL)
		{
			string snew = pch;
			splitvec.push_back(snew);
			pch = strtok (NULL, " \t");
		}

		if(splitvec.size() != 2)
		{
			cerr << "Warning: Found a row that does not represent an edge or comment." << endl;
			cerr << "Row in question: " << endl;	
			for(unsigned i=0; i<splitvec.size(); i++)
			{
				cout << splitvec[i] << endl;
			}
			exit(-1);
		}

		for(unsigned i=0; i<splitvec.size(); i++)
		{
			vertices.insert(splitvec[i]);
		}

		from.push_back(splitvec[0]);
		to.push_back(splitvec[1]);
	}
	edgelist.close();

	cout<<vertices.size()<<" "<<from.size()<<endl;
	for(int i=0;i<vertices.size();i++)
	{
		int count = 0;
		string str;
		for(int j=0;j<from.size();j++)
		{
			str = to_string(i);
			if(from[j].compare(str) == 0)
			{
				count++;
			}
		}
		cout<<count<<" ";
		for(int j=0;j<from.size();j++)
		{
			str = to_string(i);
			if(from[j].compare(str) == 0)
			{
				cout<<to[j]<<" ";
			}
		}
		cout<<endl;
	}

	return 0;
}
