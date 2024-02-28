#include <igraph.h>

int main(void) {
  igraph_integer_t num_vertices = 1000;
  igraph_integer_t num_edges = 1000;
  igraph_real_t diameter;
  igraph_t graph;

  int num_nodes = 0;
  int denom = 0;
  float total_conn = 0;
  igraph_integer_t lnc;
  igraph_integer_t anc;

  igraph_rng_seed(igraph_rng_default(), 42);

  igraph_erdos_renyi_game_gnm(
    &graph, num_vertices, num_edges,
    IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS
  );

  igraph_diameter(
    &graph, &diameter,
    /* from = */ NULL, /* to = */ NULL,
    /* vertex_path = */ NULL, /* edge_path = */ NULL,
    IGRAPH_UNDIRECTED, /* unconn= */ true
  );
  printf("Diameter of a random graph with average degree %g: %g\n",
          2.0 * igraph_ecount(&graph) / igraph_vcount(&graph),
          (double) diameter);

  num_nodes = igraph_vcount(&graph);
  for (igraph_integer_t i=0; i<num_nodes; i++) {
    for (igraph_integer_t j=i+1; j<num_nodes; j++){
      igraph_st_vertex_connectivity(&graph, &lnc, i, j, (igraph_vconn_nei_t)IGRAPH_VCONN_NEI_NEGATIVE);
      if (lnc == -1) { continue; }
      total_conn += lnc;
      denom += 1;
    }
  }
  anc = total_conn / (float) denom;
  printf("Average Node Connectivity: %f", anc);
  
  igraph_destroy(&graph);

  return 0;
}