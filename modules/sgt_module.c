#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <igraph.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>


static PyObject *ErrorObject;
static PyObject *
compute_anc(PyObject *self, PyObject *args)
{
    PyObject *result;
    int *num_cpus;
    int *allow_mp;
    char *str_adj_mat;
	igraph_t graph;

    if (!PyArg_ParseTuple(args, "sii:compute_anc", &str_adj_mat, &num_cpus, &allow_mp))
        return NULL;
	
	if ( num_cpus <= 0 || allow_mp < 0){
    	PyErr_SetString( ErrorObject, "Invalid CPU parameters.");
    	return NULL;
  	}

	igraph_adjacency(
		&graph, adj_mat, 
		IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS
	);

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
