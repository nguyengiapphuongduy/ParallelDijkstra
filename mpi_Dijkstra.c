#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
	int vertex;
	int distance;
	int path;
} element;

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

element pop(element *arr, int *size) {
	element t = arr[0];
	arr[0] = arr[*size - 1];
	heapdown(arr, --*size, 0);
	return t;
}

int main(int argc, char** argv) {

	int rank, numProc;
	int i, j, n, *graph;
	element *dlist, *local;

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
				scanf("%d", &graph[n*i + j]);
			}
		}
		dlist = (element *)malloc(n*sizeof(element));
		for (i = 0; i < n; i++) {
			dlist[i].vertex = i;
			dlist[i].distance = INT_MAX;
			dlist[i].path = -1; // nil
		}
		dlist[0].distance = 0;
	}
	MPI_Bcast(graph, n*n, MPI_INT, 0, MPI_COMM_WORLD);

	int length = n/numProc;
	local = (element *)malloc(length*sizeof(element));
	MPI_Scatter(dlist, length, dtQueueElement, local, length, dtQueueElement, 0, MPI_COMM_WORLD);

	/* Initialize to run Dijkstra */
	buildheap(local, length);
	// printf("Process %d reached the barrier, i = %d\n", rank, i);
	// MPI_Barrier(MPI_COMM_WORLD);

	for (i = 0; i < numProc; i++) {
		printf("Process %d reached the barrier, i = %d\n", rank, i);
		MPI_Barrier(MPI_COMM_WORLD);

		/* find the id of the global minimum weight */
		int localMin[2];
		localMin[0] = local[0].vertex;
		localMin[1] = local[0].distance;
		if (length == 0) {
			localMin[1] = INT_MAX;
		} // ignore localMin from a process if its queue is empty
		int globalMin[2];
		printf("Process %d send localMin v = %d, d = %d\n", rank, localMin[0], localMin[1]);
		MPI_Allreduce(localMin, globalMin, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
		printf("Process %d received globalMin v = %d, d = %d\n", rank, globalMin[0], globalMin[1]);
		for (j = 0; j < length; j++) {
			if (local[j].distance > globalMin[1] + graph[n*globalMin[0] + j]) {
				local[j].distance = globalMin[1] + graph[n*globalMin[0] + j];
				local[j].path = globalMin[0];
			}
		}
		if (globalMin[0] == localMin[0]) {
			pop(local, &length);
			printf("Process %d popped vertex %d\n", rank, localMin[0]);
		}
	}

	MPI_Gather(dlist, length, dtQueueElement, local, length, dtQueueElement, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		for (i = 0; i < n; i++) {
			printf("Vertex 0 to %d: distance = %d, previous vertex = %d\n", dlist[i].vertex, dlist[i].distance, dlist[i].path);
		}
	}

	MPI_Finalize();
	return 0;

}
