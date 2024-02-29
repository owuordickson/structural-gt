#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <igraph.h>
#include <time.h>
#include "sgt_base.h"

#define MAX_THREADS 8

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


// Function to compute Average Node Connectivity
float compute_anc(int mp) {

    igraph_t graph;
    igraph_integer_t num_vertices = 1011; // TO BE REMOVED
    igraph_integer_t num_edges = 1329; // TO BE REMOVED
    // igraph_t graph;
    igraph_integer_t num_nodes;
    igraph_integer_t count_nc = 0;
    igraph_integer_t sum_nc = 0;
    igraph_real_t anc = 0;

    igraph_rng_seed(igraph_rng_default(), 42);
    igraph_erdos_renyi_game_gnm(
        &graph, num_vertices, num_edges,
        IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS
    );
    //igraph_read_graph_edgelist(&graph, "graph.txt", 0, IGRAPH_UNDIRECTED);
    
    num_nodes = igraph_vcount(&graph);
    
    if (mp == 0){
        printf("Using single processing\n");
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
        printf("Using multiprocessing\n");
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
                pthread_create(&threads[thread_count % MAX_THREADS], NULL, compute_lnc, &args[thread_count % MAX_THREADS]);
                thread_count++;
                printf("thread %d running...\n", (thread_count % MAX_THREADS));
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

    // Destroy graph
    igraph_destroy(&graph);

    return anc;
}


// Function to convert string representation of adjacency matrix to 2D matrix
int** str_to_matrix(char* str_adj_mat, int num_vertices) {
    // Allocate memory for 2D matrix
    int** matrix = (int**)malloc(num_vertices * sizeof(int*));
    for (int i = 0; i < num_vertices; i++) {
        matrix[i] = (int*)malloc(num_vertices * sizeof(int));
    }

    // Parse string and populate matrix
    char* token = strtok(str_adj_mat, " ");
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            matrix[i][j] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    return matrix;
}

// Main function - entrypoint
int main() {
    printf("Started...\n");
    clock_t start = clock();

    float anc = compute_anc(1);
    printf("Average Node Connectivity: %f\n", anc);

    clock_t end = clock();
    double duration = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Ended ...\n");
    printf("Time: %.2f seconds\n", duration);

    return 0;
}
