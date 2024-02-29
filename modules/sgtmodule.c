#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <igraph.h>
#include <stdarg.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sgt_base.h"


static PyObject *ErrorObject;
static PyObject *
compute_anc(PyObject *self, PyObject *args)
{
    int num_cpus;
    int allow_mp;
	int length;
    char *str_adj_mat;

    if (!PyArg_ParseTuple(args, "siii:compute_anc", &str_adj_mat, &length, &num_cpus, &allow_mp))
        return NULL;
	
	if ( num_cpus <= 0 || allow_mp < 0){
    	PyErr_SetString( ErrorObject, "Invalid CPU parameters.");
    	return NULL;
  	}

    // Declare required variables
    const int MAX_THREADS = num_cpus;
	int size;
	igraph_matrix_t adj_mat;

	igraph_t graph;
	igraph_integer_t num_nodes;
    igraph_integer_t count_nc = 0;
    igraph_integer_t sum_nc = 0;
    igraph_real_t anc = 0;

	// Get size of adjacency matrix
	size = (int)sqrt((double)(length));

	// Convert string to matrix
    adj_mat = str_to_matrix(str_adj_mat, size);

    // Build igraph
	/*igraph_adjacency(
		&graph, &adj_mat,
		IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS
	);

	num_nodes = igraph_vcount(&graph);
	if (allow_mp == 0){
        //printf("Using single processing\n");
        igraph_integer_t lnc;
        for (igraph_integer_t i=0; i<num_nodes; i++) {
            for (igraph_integer_t j=i+1; j<num_nodes; j++){
                igraph_st_vertex_connectivity(&graph, &lnc, i, j, (igraph_vconn_nei_t)IGRAPH_VCONN_NEI_NEGATIVE);
                if (lnc == -1) { continue; }
                sum_nc += lnc;
                count_nc += 1;
            }
        }

    }
    else {
        //printf("Using multiprocessing\n");
        // Initialize mutex
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, NULL);

        // Create thread pool
        pthread_t threads[MAX_THREADS];
        ThreadArgsLNC args[MAX_THREADS];
        int thread_count = 0;

        // Initialize thread pool
        for (int i = 0; i < MAX_THREADS; i++) {
            args[i].graph = &graph;
            args[i].mutex = &mutex;
            args[i].total_nc = &sum_nc;
            args[i].total_count = &count_nc;
        }

        // Create threads for computing LNC
        for (igraph_integer_t i = 0; i < num_nodes; i++) {
            for (igraph_integer_t j = i + 1; j < num_nodes; j++) {
                if (thread_count >= MAX_THREADS) {
                    // Wait for a thread to finish before starting a new one
                    pthread_join(threads[thread_count % MAX_THREADS], NULL);
                    thread_count++;
                }
                args[thread_count % MAX_THREADS].i = i;
                args[thread_count % MAX_THREADS].j = j;
                pthread_create(&threads[thread_count % MAX_THREADS], NULL, &compute_lnc, &args[thread_count % MAX_THREADS]);
                thread_count++;
                //printf("thread %d running...\n", (thread_count % MAX_THREADS));
            }
        }

        // Join threads
        for (int i = 0; i < MAX_THREADS && i < thread_count; i++) {
            pthread_join(threads[i], NULL);
        }

        // Destroy mutex
        pthread_mutex_destroy(&mutex);
    }

    // Compute ANC
    anc = (float) sum_nc / count_nc;
    */
	// Print the matrix
    printf("Adjacency Matrix:\n");
    for (igraph_integer_t i = 0; i < size; i++) {
        for (igraph_integer_t j = 0; j < size; j++) {
            printf("%f ", MATRIX(adj_mat, i, j));
        }
        printf("\n");
    }

    // Destroy graph
    igraph_matrix_destroy(&adj_mat);
    igraph_destroy(&graph);

    return PyFloat_FromDouble((double) anc);
}
static char compute_anc_doc[] =
"A C method that uses iGraph library to compute average node connectivity of a graph.\n"
"\n"
"Arguments:\n"
"   A       string      adjacency matrix of graph\n"
"   l       int         number of values/items in the variable A\n"
"   cpus    int         number of available CPUs\n"
"   mp      int         allow multi-processing (0: No, 1: Yes)\n"
"\n"
"The length of the graph should be a squared(N) since an adjacency matrix is NxN in size.\n"
"Returns the Average Node Connectivity as a float value.\n";


static char sgt_doc[] =
"A C module that uses iGraph library to compute GT metrics.\n"
"\n";

/* Method Table: ist of functions defined in the module */
static PyMethodDef sgt_methods[] = {
    {"compute_anc", compute_anc, METH_VARARGS, compute_anc_doc },
    //{"compute_lnc", compute_lnc, METH_VARARGS, "Compute local node connectivity." },
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Create module */
static struct PyModuleDef sgtmodule = {
    PyModuleDef_HEAD_INIT,
    "sgt",   /* name of module */
    sgt_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    sgt_methods
};

/* Initialization function for the module */
PyMODINIT_FUNC
PyInit_sgt(void)
{
    PyObject *m;

    m = PyModule_Create(&sgtmodule);
    if (m == NULL)
        return NULL;

    ErrorObject = PyErr_NewException("sgt.error", NULL, NULL);
    Py_XINCREF(ErrorObject);
    if (PyModule_AddObject(m, "error", ErrorObject) < 0) {
        Py_XDECREF(ErrorObject);
        Py_CLEAR(ErrorObject);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

