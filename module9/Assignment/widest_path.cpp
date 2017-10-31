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
#define VERTEX_NUMSETS 2
#define EDGE_NUMSETS 1

#define MAX_INT 10

int widest_path_sub()
{
    int i = 0;
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
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

    // Fill graph variables
    weights_h = (float*)malloc(NUM_EDGES*sizeof(float));
    source_indices_h = (int*) malloc(NUM_EDGES*sizeof(int));

    for(i = 0; i < NUM_EDGES; i++){
      weights_h[i] = (float) (rand() / MAX_INT);
      printf("weights[%d] = %0.1f\n", i, weights_h[i]);
    }
    for(i = 0; i < NUM_EDGES; i++) {
      source_indices_h[i] = rand() % NUM_VERTICES;
      printf("source indices[%d] = %d\n", i, source_indices_h[i]);
    }

    destination_offsets_h = (int*) malloc((NUM_VERTICES+1)*sizeof(int));
    bookmark_h = (float*)malloc(NUM_VERTICES*sizeof(float));

    for(i = 0; i < NUM_VERTICES+1; i++) {
      destination_offsets_h[i] = rand() % NUM_VERTICES;
      printf("destination offset[%d] = %d", i, destination_offsets_h[i]);
      bookmark_h[i] = (float) (rand() / MAX_INT);
    }
    for(i = 0; i < NUM_VERTICES; i++) {
      bookmark_h[i] = (float) (rand() / MAX_INT);
      printf("bookmark[%d] = %0.1f", i, bookmark_h[i]);
    }

    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Host variables for result
    float *pr_1;
    void** vertex_dim;
    pr_1 = (float*)malloc(NUM_VERTICES*sizeof(float));
    vertex_dim = (void**)malloc(VERTEX_NUMSETS*sizeof(void*));
    vertex_dim[0] = (void*)bookmark_h;
    vertex_dim[1]= (void*)pr_1;

    // Device variables for result
    cudaDataType_t d_edge_dim = CUDA_R_32F;
    cudaDataType_t *d_vertex_dim = (cudaDataType_t*)malloc(VERTEX_NUMSETS*sizeof(cudaDataType_t));
    d_vertex_dim[0] = CUDA_R_32F;
    d_vertex_dim[1] = CUDA_R_32F;


    cudaEvent_t start, stop;
    float duration;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Recording from load to copy back
    cudaEventRecord(start, 0);

    int ret = nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32);
    ret += nvgraphAllocateVertexData(handle, graph, VERTEX_NUMSETS, d_vertex_dim);
    ret += nvgraphAllocateEdgeData(handle, graph, EDGE_NUMSETS, &d_edge_dim);
    if(ret != 0 ) {
      printf("Failed to set up graph or allocate memory for graph data\n");
      return EXIT_FAILURE;
    }
    nvgraphSetVertexData(handle, graph, vertex_dim[1], 1);
    nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0);

    nvgraphSssp(handle, graph, 0, 0, 0);

    // Get and print result
    nvgraphGetVertexData(handle, graph, vertex_dim[1], 1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    printf("Elapsed Time: %f\n", duration);
    for (i = 0; i < NUM_VERTICES; i++) {
      printf("%f\n",pr_1[i]);
    }

    // Clean Up
    nvgraphDestroyGraphDescr(handle, graph);
    nvgraphDestroy(handle);

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(vertex_dim);
    free(d_vertex_dim);
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
