#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <igraph.h>
#include <stdarg.h>
#include "sgt_base.h"

// Function to compute Local Node Connectivity
void* compute_lnc(void *arg) {
    ThreadArgsLNC *args = (ThreadArgsLNC*)arg;
    igraph_integer_t lnc;

    igraph_st_vertex_connectivity(args->graph, &lnc, args->i, args->j, IGRAPH_VCONN_NEI_NEGATIVE);

    // Update shared data under mutex lock
    pthread_mutex_lock(args->mutex);
    if (lnc != -1){
        *(args->total_nc) += lnc;
        *(args->total_count) += 1;
        //printf("got %d\n", lnc);
        //printf("NC:%d Count:%d \n", *(args->total_nc), *(args->total_count));
    }
    pthread_mutex_unlock(args->mutex);
    
    pthread_exit(NULL);
}

// Function to convert string representation of adjacency matrix to 2D matrix
igraph_matrix_t str_to_matrix(char* str_adj_mat, igraph_integer_t num_vertices) {
    igraph_matrix_t mat;
    igraph_matrix_init(&mat, num_vertices, num_vertices);

    // Parse string and populate matrix
    char* token = strtok(str_adj_mat, " ");
    for (igraph_integer_t i = 0; i < num_vertices; i++) {
        for (igraph_integer_t j = 0; j < num_vertices; j++) {
            MATRIX(mat, i, j) = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    return mat;
}

