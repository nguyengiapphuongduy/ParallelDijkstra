/* Parallel Dijkstra single source shortest-path algorithm

@author: 1610473 Nguyen Giap Phuong Duy

compile:
mpicc -g -Wall -o mpi_Dijkstra mpi_Dijkstra.c

execute:
mpiexec -n 2 ./mpi_Dijkstra < input.txt

INPUT: text file
line 1: number of vertices (n)
line 2 to n + 1: n integer represent the adjacency matrix,
INFINITY if no edge. Example:

4
0 INFINITY 2 INFINITY
1 0 INFINITY INFINITY
2 3 0 INFINITY
INFINITY INFINITY INFINITY 0

 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 1073741823 /* = (2^30 - 1) ~ INT_MAX/2 */
#define NIL -1
#define ZERONIL -2

typedef struct {
	int vertex;
	int distance;
	int path;
} element;


/* Heap definitions */
void heapdown(element *arr, int size, int idx) {
	int next;
	while (2*idx + 1 < size) {
		if ((2*idx + 2 < size) && (arr[2*idx + 1].distance > arr[2*idx + 2].distance)) {
			next = 2*idx + 2;
		} else {
			next = 2*idx + 1;
		}
		if (arr[idx].distance > arr[next].distance) {
			element t = arr[idx];
			arr[idx] = arr[next];
			arr[next] = t;
		}
		idx = next;
	}
}

void buildheap(element *arr, int size) {
	int i;
	for (i = size - 1; i >= 0; i--) {
		heapdown(arr, size, i);
	}
}

void pop(element *arr, int *size) {
	if (*size > 0) {
		element t = arr[0];
		arr[0] = arr[*size - 1];
		arr[*size - 1] = t;
		heapdown(arr, --*size, 0);
	}
}
/* End of heap definitions */

int printpath(element* arr, int v) {
	if (v < 0) {
		if (v == ZERONIL) return 1;
		else return 0;
	}
	int t = printpath(arr, arr[v].path);
	if (t == 1) printf(" %d", v);
	return t;
}


int main(int argc, char** argv) {

	int rank, numProc;
	int i, j, n, *graph;
	element *dlist, *local, *queue;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);

	MPI_Datatype dtQueueElement;
	MPI_Type_contiguous(3, MPI_INT, &dtQueueElement);
	MPI_Type_commit(&dtQueueElement);

	/* this step is to ensure that every processor knows the value of n */
	if (rank == 0) {
		scanf("%d", &n);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	/* read the adjacency matrix and then initialize the distance list */
	graph = (int *)malloc(n*n*sizeof(int));
	if (rank == 0) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				char *r = (char *)malloc(64*sizeof(char));
				scanf("%s", r);
				if (strcmp(r, "INFINITY") == 0) {
					graph[n*i + j] = INF;
				} else {
					graph[n*i + j] = atoi(r);
				}
			}
		}
		dlist = (element *)malloc(n*sizeof(element));
		for (i = 0; i < n; i++) {
			dlist[i].vertex = i;
			dlist[i].distance = INF;
			dlist[i].path = NIL;
		}
		dlist[0].distance = 0;
		dlist[0].path = ZERONIL;
	}
	MPI_Bcast(graph, n*n, MPI_INT, 0, MPI_COMM_WORLD);

	int length = n/numProc;
	local = (element *)malloc(length*sizeof(element));
	queue = (element *)malloc(length*sizeof(element));
	MPI_Scatter(dlist, length, dtQueueElement, local, length, dtQueueElement, 0, MPI_COMM_WORLD);
	memcpy(queue, local, length*sizeof(element));

	/* Initialization to run Dijkstra */
	int offset = local[0].vertex;
	buildheap(queue, length);

	for (i = 0; i < n; i++) {
		// MPI_Barrier(MPI_COMM_WORLD);

		/* find the id of the global minimum weight */
		int queueMin[2]; // struct {distance, vertex}
		queueMin[0] = queue[0].distance;
		queueMin[1] = queue[0].vertex;
		if (length == 0) {
			queueMin[0] = INF + 1;
			queueMin[1] = NIL;
		} // ignore queueMin from a process if its queue is empty

		int globalMin[2]; // struct {distance, vertex}
		MPI_Allreduce(queueMin, globalMin, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

		/* Dijkstra Algorithm as normal */
		for (j = 0; j < length; j++) {
			if (queue[j].distance > globalMin[0] + graph[n*globalMin[1] + queue[j].vertex]) {
				queue[j].distance = globalMin[0] + graph[n*globalMin[1] + queue[j].vertex];
				queue[j].path = globalMin[1];
			}
		}
		if (globalMin[1] == queueMin[1]) {
			pop(queue, &length);
		}
	}

	/* Parse the queue (suffered indices) to the local (arranged indices) */
	length = n/numProc;
	for (i = 0; i < length; i++) {
		int idx = queue[i].vertex - offset;
		local[idx].distance = queue[i].distance;
		local[idx].path = queue[i].path;
	}

	/* Gather and print result */
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(local, length, dtQueueElement, dlist, length, dtQueueElement, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("Shortest path from vertex 0 to:\n");
		for (i = 1; i < n; i++) {
			printf("%d: distance = ", dlist[i].vertex);
			if (dlist[i].distance >= INF) printf("INFINITY");
			else printf("%d", dlist[i].distance);
			printf(", path =");
			int r = printpath(dlist, i);
			if (r == 1) printf("\n");
			else printf(" NOTFOUND\n");
		}
	}

	MPI_Finalize();
	return 0;

}
