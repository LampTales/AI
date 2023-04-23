import argparse
import numpy as np
import sys
import heapq


def get_args():
    parser = argparse.ArgumentParser(description='CARP')
    parser.add_argument('file_path', type=str, help='file path')
    parser.add_argument('-t', '--termination', type=int, help='termination')
    parser.add_argument('-s', '--rand_seed', type=int, help='random seed')
    args = parser.parse_args()
    return args


def read_data(data_path):
    file = open(data_path, 'r')
    file.readline()  # skip the name
    header = dict()
    edges = []
    for _ in range(7):
        line = file.readline()
        data = line.split(':')
        header[data[0].strip().lower()] = int(data[1].strip())
    # print(header)
    file.readline()  # skip the header line
    line = file.readline()
    while line.strip() != 'END':
        data = line.split()
        edges.append(((int(data[0]), int(data[1])), int(data[2]), int(data[3])))
        line = file.readline()
    # print(edges)
    return header, edges


def deal_edges(n, edges):
    clean_list = []
    matrix = np.full((n, n), np.inf)
    for i in range(n):
        matrix[i][i] = 0
    # load edges
    for edge in edges:
        if edge[3] != 0:
            clean_list.append(edge)
        if matrix[edge[0][0]][edge[0][1]] > edge[2]:
            matrix[edge[0][0]][edge[0][1]] = edge[2]
            matrix[edge[0][1]][edge[0][0]] = edge[2]

    distance = np.array(matrix)
    # do Floyd
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    return clean_list, matrix, distance


def find_edge(cur, cur_cap, clean_list, distance):

    return None



def main():
    args = get_args()
    print(args)
    header, edges = read_data(args.file_path)
    n, root, cap, vehicles = header.get('vertices'), header.get('depot'), header.get('capacity'), header.get('vehicles')
    clean_list, matrix, distance = deal_edges(n, edges)
    path_list = []
    while len(clean_list) != 0:
        path = []
        cur = root
        cur_cap = cap
        edge = find_edge(cur, cur_cap, clean_list, distance)



if __name__ == "__main__":
    main()
