/*
 * Assignment 09 C
 * Sarah Helble
 * 30 Oct 2017
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

#define NUM_VERTICES 10
#define NUM_EDGES 15
#define VERTEX_NUMSETS 3
#define EDGE_NUMSETS 1

#define MAX_INT 10

int widest_path_sub()
{
    int i = 0;
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    int *destination_offsets_h, *source_indices_h;
    float *weights_h, *bookmark_h;

    if(nvgraphCreate (&handle) != 0) {
      printf("Failed to create graph handle\n");
      return EXIT_FAILURE;
    }
    if(nvgraphCreateGraphDescr (handle, &graph) != 0) {
      printf("Failed to create graph");
      return EXIT_FAILURE;
    }

    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    CSC_input->nvertices = NUM_VERTICES;
    CSC_input->nedges = NUM_EDGES;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    //TODO: Not sure about this whole part
    float *pr_1,*pr_2;
    void** vertex_dim;
    cudaDataType_t* vertex_dimT;
    pr_1 = (float*)malloc(NUM_VERTICES*sizeof(float));
    pr_2 = (float*)malloc(NUM_VERTICES*sizeof(float));
    vertex_dim = (void**)malloc(VERTEX_NUMSETS*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(VERTEX_NUMSETS*sizeof(cudaDataType_t));

    // Initialize host data
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]= CUDA_R_32F;
    //////////////////////

    //Fill graph variables
    weights_h = (float*)malloc(NUM_EDGES*sizeof(float));
    source_indices_h = (int*) malloc(NUM_EDGES*sizeof(int));

    for(i = 0; i < NUM_EDGES; i++){
      weights_h[i] = (float) (rand() / MAX_INT);
      source_indices_h[i] = rand() % NUM_VERTICES;
    }

    destination_offsets_h = (int*) malloc((NUM_VERTICES+1)*sizeof(int));
    bookmark_h = (float*)malloc(NUM_VERTICES*sizeof(float));

    for(i = 0; i < NUM_VERTICES; i++) {
      destination_offsets_h[i] = rand() % NUM_VERTICES;
      bookmark_h[i] = (float) (rand() / MAX_INT);
    }
    destination_offsets_h[i] = rand() % NUM_VERTICES;

    vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1, vertex_dim[2]= (void*)pr_2;

    int ret = nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32);
    ret += nvgraphAllocateVertexData(handle, graph, VERTEX_NUMSETS, vertex_dimT);
    ret += nvgraphAllocateEdgeData(handle, graph, EDGE_NUMSETS, &edge_dimT);
    if(ret != 0 ) {
      printf("Failed to set up graph or allocate memory for graph data\n");
      return EXIT_FAILURE;
    }

    nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0);

    // First run with default values
    nvgraphWidestPath(handle, graph, 0, 0, 1);

    // Get and print result
    nvgraphGetVertexData(handle, graph, vertex_dim[1], 1);
    for (i = 0; i<n; i++) {
      printf("%f\n",pr_1[i]);
    }

    //Clean
    nvgraphDestroyGraphDescr(handle, graph);
    nvgraphDestroy(handle);

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(pr_2);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    return EXIT_SUCCESS;
}

int main(void)
{
  printf("\nRun 1 of widest_path function:\n");
  widest_path_sub();

  printf("\nRun 2 of widest_path function:\n");
  widest_path_sub();
}
