import argparse
import random
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


vary_rate = 0.1
group_size = 300
select_size = 5
select_range = range(select_size)


def cal_length(root, path, distance, matrix):
    cur = root
    length = 0
    for edge in path:
        length += distance[cur][edge[0]] + matrix[edge[0]][edge[1]]
        cur = edge[1]
    length += distance[path[-1][1]][root]
    return length


def cal_total_length(path_list):
    total = 0
    for _, path_len in path_list:
        total += path_len
    return total


def copy_list(path_list):
    new_list = []
    for path, path_len in path_list:
        new_path = []
        for edge in path:
            new_path.append(edge)
        new_list.append((new_path, path_len))
    return new_list


def generate(group):
    raw_list = []
    list1 = (random.choice(group))[1]
    list2 = (random.choice(group))[1]
    for i in range(len(list1)):
        path = []
        if random.random() < 0.5:
            for edge in list1[i][0]:
                path.append(edge)
        else:
            for edge in list2[i][0]:
                path.append(edge)
        raw_list.append(path)
    return raw_list


def vary(raw_list, root, distance, matrix):
    com_list = []
    for path in raw_list:
        if random.random() < vary_rate:
            if random.random() < 0.5:
                # kind 1 vary
                spot1 = random.choice(range(len(path)))
                spot2 = random.choice(range(len(path)))
                temp = path[spot1]
                path[spot1] = path[spot2]
                path[spot2] = temp
            else:
                # kind 2 vary
                spot = random.choice(range(len(path)))
                path[spot] = (path[spot][1], path[spot][0])
        com_list.append((path, cal_length(root, path, distance, matrix)))
    return com_list, cal_total_length(com_list)


def optimize(n, root, path_list, distance, matrix, time_start, time_limit):
    select_path_list = path_list
    select_len = cal_total_length(path_list)

    # init
    group = []
    for _ in range(group_size):
        group.append((select_len, copy_list(select_path_list)))

    while time.time() - time_start < time_limit - 10:
        next_group = []
        for _ in range(group_size):
            new_list, total_len = vary(generate(group), root, distance, matrix)
            next_group.append((total_len, new_list))
        next_group.sort()
        group = next_group[0:5]
        select_path_list = group[0][1]
        select_len = group[0][0]
        print("round select len: " + str(select_len))

    return select_path_list, select_len


if __name__ == "__main__":
    main()



