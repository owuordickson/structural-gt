#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <igraph.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sgt_base.h"


static PyObject *ErrorObject;
static PyObject *
compute_anc(PyObject *self, PyObject *args)
{
    PyObject *result;
    int *num_cpus;
    int *allow_mp;
	int *length;
    char *str_adj_mat;
	igraph_t graph;
	int size;
	int** adj_mat;

    if (!PyArg_ParseTuple(args, "siii:compute_anc", &str_adj_mat, &length, &num_cpus, &allow_mp))
        return NULL;
	
	if ( num_cpus <= 0 || allow_mp < 0){
    	PyErr_SetString( ErrorObject, "Invalid CPU parameters.");
    	return NULL;
  	}

	// Get size of adjascency matrix
	size = (int)sqrt((double)(*length));

	// Convert string to matrix
    adj_mat = str_to_matrix(str_adj_mat, size);

	igraph_adjacency(
		&graph, adj_mat, 
		IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS
	);



	// Print the matrix
    printf("Adjacency Matrix:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", adj_mat[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < size; i++) {
        free(adj_mat[i]);
    }
    free(adj_mat);

}
static char compute_anc_doc[] =
"\n"
"\n"
"Arguments:\n"
"\n"
"\n"
"\n";


/* List of functions defined in the module */
static PyMethodDef metric_methods[] = {
    {"compute_anc",    compute_anc,     METH_VARARGS, compute_anc_doc },
	{NULL,		NULL}		/* sentinel */
};

/* Initialization function for the module */
DL_EXPORT(void)
init_metric(void)
{
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule("gt_metrics", metric_methods);

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyErr_NewException("gt_metrics.error", NULL, NULL);
	PyDict_SetItemString(d, "error", ErrorObject);
}
