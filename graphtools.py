import numpy as np
import networkx as nx
from tqdm import tqdm
from visualization import *
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

import metis


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def knn_graph(df, k, verbose=False):
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    if verbose:
        print("Building kNN graph (k = %d)..." % (k))
    iterpoints = tqdm(enumerate(points), total=len(
        points)) if verbose else enumerate(points)
    for i, p in iterpoints:
        distances = list(map(lambda x: euclidean_distance(p, x), points))
        closests = np.argsort(distances)[1:k+1]  # second trough kth closest
        # print(distances[0])
        for c in closests:
            g.add_edge(i, c, weight=1.0 / distances[c], similarity=int(
                1.0 / distances[c] * 1e4))
        g.nodes[i]['pos'] = p
    g.graph['edge_weight_attr'] = 'similarity'
    return g


def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype='cut', ufactor=250)
    # print(edgecuts)
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]['cluster'] = parts[i]
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def pre_part_graph(graph, k, df=None, verbose=False):
    if verbose:
        print("Begin clustering...")
    clusters = 0
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]['cluster'] = 0
    cnts = {}
    cnts[0] = len(graph.nodes())

    while clusters < k - 1:
        maxc = -1
        maxcnt = 0
        for key, val in cnts.items():
            if val > maxcnt:
                maxcnt = val
                maxc = key
        s_nodes = [n for n in graph.nodes if graph.nodes[n]['cluster'] == maxc]
        s_graph = graph.subgraph(s_nodes)
        edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype='cut', ufactor=250)
        new_part_cnt = 0
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == 1:
                graph.nodes[p]['cluster'] = clusters + 1
                new_part_cnt = new_part_cnt + 1
        cnts[maxc] = cnts[maxc] - new_part_cnt
        cnts[clusters + 1] = new_part_cnt
        clusters = clusters + 1

    edgecuts, parts = metis.part_graph(graph, k)
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def get_cluster(graph, clusters):
    nodes = [n for n in graph.nodes if graph.nodes[n]['cluster'] in clusters]
    return nodes


def connecting_edges(partitions, graph):
    cut_set = []
    #print (partitions[0])
    #print (partitions[1])
    #print('next iteration')
    #adj_graph = nx.to_pandas_adjacency(graph)
    #print(adj_graph)
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    #print (cut_set)
    return cut_set


def cuda_connecting_edges(partitions, graph):
    block = (len(partitions[0]),1,1)
    grid = (1,1)
    
    mod = source_module("""
    __global__ void connecting_edges(float* dest, float* first_cluster, float* second_cluster, 
                                        bool* adj_matrix, int second_cluster_length, int matrix_block_size)
    {
        int set_index = threadId.x;
        int return_index = threadId.x;
        
        for(int second_index = 0; second_index < second_cluster_length; second_index++ )
            {
                if(adj_matrix[ (set_index * matrix_block_size) + second_index ] > 0 )
                    dest[return_index] = { {first_cluster[set_index]}, {second_cluster[second_index]} };
                else 
                    dest[return_index] = { {-1}, {-1} };
            } 
    }  
    """)   
    
    connecting_edges = mod.get_function('connecting_edges')
    
    return_set = [0] * ( len(partitions[0]) * len(partitions[1]) )   
    return_set = return_set.as_type(numpy.float32))
    gpu_return_set = cuda.mem_alloc(return_set.nbytes)
    cuda.memcpy_htod(gpu_return_set, return_set)
    
    cluster_i = partitions[0].nodes
    cluster_i = cluster_i.as_type(numpy.float32))
    gpu_cluster_i = cuda.mem_alloc(return_set.nbytes)
    cuda.memcpy_htod(gpu_cluster_i, cluster_i)
    
    cluster_j = partitions[1].nodes
    cluster_j = cluster_j.as_type(numpy.float32))
    gpu_cluster_j = cuda.mem_alloc(return_set.nbytes)
    cuda.memcpy_htod(gpu_cluster_j, cluster_j)
    
    second_cluster_length = len(cluster_j)
    
    graph = nx.to_pandas_adjacency(graph)
    
    list_graph = []
    for i in graph
        for j in graph  
            list_graph.adj_graph[i][j]
        
    gpu_adj_matrix = gpu_array.to_gpu(list_graph)
    
    matrix_block_size = len(graph)
    
    connecting_edges( drv.out(gpu_return_set), drv.in(gpu_cluster_i), drv.in(gpu_cluster_j), drv.in(gpu_adj_matrix), 
                        drv.in(second_cluster_length), driv.in(matrix_block_size) block, grid )
    
    pair_set = numpy.empty_like(return_set)
    cuda.memcpy_dtoh(pair_set, gpu_return_set)
    
    pair_set = [pair_set for pair_set in a if pair_set != (-1,-1)]
    
    return pair_set


def min_cut_bisector(graph):
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
    return connecting_edges(partitions, graph)


def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]


def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights
