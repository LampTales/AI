import argparse
import time
import numpy as np


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
        edges.append(((int(data[0]) - 1, int(data[1]) - 1), int(data[2]), int(data[3])))
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
        if edge[2] != 0:
            clean_list.append(edge)
        if matrix[edge[0][0]][edge[0][1]] > edge[1]:
            matrix[edge[0][0]][edge[0][1]] = edge[1]
            matrix[edge[0][1]][edge[0][0]] = edge[1]

    distance = np.array(matrix)
    # do Floyd
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    return clean_list, matrix, distance


def find_edge(cur, cur_cap, clean_list, distance, root, cap_des):
    start = -1
    select_index = -1
    select_edge = None
    min_cost = np.inf
    for i in range(len(clean_list)):
        edge = clean_list[i]
        if cur_cap >= edge[2]:
            this_min, this_start = (distance[cur][edge[0][0]], edge[0][0]) \
                if distance[cur][edge[0][0]] < distance[cur][edge[0][1]] \
                else (distance[cur][edge[0][1]], edge[0][1])
            if this_min < min_cost:
                start, select_index, select_edge, min_cost = this_start, i, edge, this_min
            elif this_min == min_cost:
                if cap_des and distance[root][this_start] > distance[root][start]:
                    start, select_index, select_edge, min_cost = this_start, i, edge, this_min
                elif (not cap_des) and distance[root][this_start] < distance[root][start]:
                    start, select_index, select_edge, min_cost = this_start, i, edge, this_min
    return start, select_edge, select_index


def main():
    time_start = time.time()
    args = get_args()
    print(args)
    header, edges = read_data(args.file_path)
    n, root, cap, vehicles = header.get('vertices'), header.get('depot') - 1, header.get('capacity'), header.get('vehicles')
    clean_list, matrix, distance = deal_edges(n, edges)

    path_list = []
    while len(clean_list) != 0:
        # for each path
        path = []
        path_len = 0
        cur = root
        cur_cap = cap
        while True:
            start, edge, index = find_edge(cur, cur_cap, clean_list, distance, root, cur_cap > cap / 2)
            if edge is None:
                break
            # print(edge)
            path.append((edge[0][0], edge[0][1]) if start == edge[0][0] else (edge[0][1], edge[0][0]))
            path_len += distance[cur][start] + edge[1]
            cur = path[-1][1]
            cur_cap -= edge[2]
            clean_list.pop(index)
        path_len += distance[cur][root]
        path_list.append((path, path_len))

    path_list, _ = optimize(n, root, path_list, distance, matrix, time_start, args.termination)

    time_end = time.time()
    print("time taken: " + str(time_end - time_start))

    # output
    print('s ', end='')
    total_len = 0
    path_cnt = 0
    for path, path_len in path_list:
        path_cnt += 1
        total_len += path_len
        print(0, end=',')
        for edge in path:
            print((edge[0] + 1, edge[1] + 1), end=',')
        print(0, end='') if path_cnt == len(path_list) else print(0, end=',')
    print()
    print('q ' + str(total_len))

    time_end = time.time()
    print("time taken: " + str(time_end - time_start))

    return


if __name__ == "__main__":
    main()


def cal_length(root, path, distance, matrix):
    cur = root
    length = 0
    for edge in path:
        length += distance[cur][edge[0]] + matrix


def optimize(n, root, path_list, distance, matrix, time_start, time_limit):
    select_path_list = path_list
    total_len = np.inf
    return select_path_list, total_len



