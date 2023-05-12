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
    # print(file.readline())  # skip the name
    file.readline()
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
    demand = np.full((n, n), 0)
    for i in range(n):
        matrix[i][i] = 0
    # load edges
    for edge in edges:
        if edge[2] != 0:
            clean_list.append(edge)
            demand[edge[0][0]][edge[0][1]] = edge[2]
            demand[edge[0][1]][edge[0][0]] = edge[2]

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
    return clean_list, matrix, distance, demand


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
    random.seed(args.rand_seed)
    # print(args)
    header, edges = read_data(args.file_path)
    n, root, cap, vehicles = header.get('vertices'), header.get('depot') - 1, header.get('capacity'), header.get(
        'vehicles')
    clean_list, matrix, distance, demand = deal_edges(n, edges)

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

    path_list, _ = optimize(n, root, path_list, distance, matrix, time_start, args.termination, demand, cap)

    # test_list = [([(1, 8), (8, 3), (3, 2), (2, 1), (1, 9), (9, 3), (3, 4), (4, 11)], 0),
    #              ([(1, 10), (10, 9), (9, 8), (8, 12), (12, 10), (12, 11), (11, 1)], 0),
    #              ([(1, 4), (4, 2), (2, 5), (5, 7), (7, 2)], 0), ([(5, 6), (11, 6), (6, 4), (4, 7), (5, 1)], 0)]
    # print(legal(test_list, demand, cap))

    time_end = time.time()
    # print("time taken: " + str(time_end - time_start))

    # output
    print('s ', end='')
    total_len = 0
    path_cnt = 0
    for path, path_len in path_list:
        path_cnt += 1
        total_len += path_len
        print(0, end=',')
        for edge in path:
            print('(' + str(int(edge[0] + 1)) + ',' + str(int(edge[1] + 1)) + ')', end=',')
        print(0, end='') if path_cnt == len(path_list) else print(0, end=',')
    print()
    print('q ' + str(int(total_len)))

    time_end = time.time()
    # print("time taken: " + str(time_end - time_start))

    return


def cal_length(path, root, distance, matrix):
    cur = root
    length = 0
    for edge in path:
        length += distance[cur][edge[0]] + matrix[edge[0]][edge[1]]
        cur = edge[1]
    length += distance[path[-1][1]][root]
    return length


def cal_total_length(path_list, root, distance, matrix):
    total = 0
    i = 0
    while i < len(path_list):
        # print(path_list[i][0])
        if len(path_list[i][0]) == 0:
            path_list.pop(i)
            # print("remove!")
            continue
        length = cal_length(path_list[i][0], root, distance, matrix)
        path_list[i] = (path_list[i][0], length)
        total += length
        i += 1
    return total


def copy_list(path_list):
    new_list = []
    for path, path_len in path_list:
        new_path = []
        for edge in path:
            new_path.append(edge)
        new_list.append((new_path, path_len))
    return new_list


def legal(path_list, demand, cap):
    for path, _ in path_list:
        de = 0
        for edge in path:
            de += demand[edge[0]][edge[1]]
        if de > cap:
            # print("false: " + str(de))
            return False
    return True


def vary(path_list):
    new_list = copy_list(path_list)
    opcode = random.choice(range(4))
    inverse = random.choice(range(2))
    inverse_p = random.choice(range(2))

    # insertion 1
    if opcode == 0:
        path = new_list[random.choice(range(len(new_list)))][0]
        pos = random.choice(range(len(path)))
        select = path[pos]
        path.pop(pos)
        new_path = random.choice(range(len(new_list) + 1))
        if new_path == len(new_list):
            if inverse:
                new_list.append(([(select[1], select[0])], 0))
            else:
                new_list.append(([select], 0))
        else:
            if inverse:
                new_list[new_path][0].insert(random.choice(range(len(new_list[new_path][0]) + 1)),
                                             (select[1], select[0]))
            else:
                new_list[new_path][0].insert(random.choice(range(len(new_list[new_path][0]) + 1)), select)

    # insertion 2
    elif opcode == 1:
        path = new_list[random.choice(range(len(new_list)))][0]
        if len(path) < 2:
            return None
        pos = random.choice(range(len(path) - 1))
        select = [path[pos], path[pos + 1]]
        path.pop(pos)
        path.pop(pos)
        new_path = random.choice(range(len(new_list) + 1))
        if new_path == len(new_list):
            if inverse:
                new_list.append(([(select[1][1], select[1][0]), (select[0][1], select[0][0])], 0))
            else:
                new_list.append((select, 0))
        else:
            if inverse:
                pos = random.choice(range(len(new_list[new_path][0]) + 1))
                new_list[new_path][0].insert(pos, (select[0][1], select[0][0]))
                new_list[new_path][0].insert(pos + 1, (select[1][1], select[1][0]))
            else:
                pos = random.choice(range(len(new_list[new_path][0]) + 1))
                new_list[new_path][0].insert(pos, select[1])
                new_list[new_path][0].insert(pos + 1, select[0])

    # swap
    elif opcode == 2:
        pos1 = random.choice(range(len(new_list)))
        pos1 = (pos1, random.choice(range(len(new_list[pos1][0]))))
        pos2 = random.choice(range(len(new_list)))
        pos2 = (pos2, random.choice(range(len(new_list[pos2][0]))))
        temp = (new_list[pos1[0]][0])[pos1[1]]
        if inverse_p:
            (new_list[pos1[0]][0])[pos1[1]] = ((new_list[pos2[0]][0])[pos2[1]][1], (new_list[pos2[0]][0])[pos2[1]][0])
        else:
            (new_list[pos1[0]][0])[pos1[1]] = (new_list[pos2[0]][0])[pos2[1]]
        if inverse:
            (new_list[pos2[0]][0])[pos2[1]] = (temp[1], temp[0])
        else:
            (new_list[pos2[0]][0])[pos2[1]] = temp

    #
    elif opcode == 3:
        path = new_list[random.choice(range(len(new_list)))][0]
        op_len = random.choice(range(len(path))) + 1
        pos = random.choice(range(len(path) - op_len + 1))
        sub_path = path[pos: pos + op_len]
        for i in range(op_len):
            path[pos + i] = (sub_path[op_len - 1 - i][1], sub_path[op_len - 1 - i][0])

    return new_list


group_size = 300

select_size = 8
local_times = 160
static_bare = 6
group_range = range(group_size)
select_range = range(select_size)


def optimize(n, root, path_list, distance, matrix, time_start, time_limit, demand, cap):
    select_path_list = path_list
    select_len = cal_total_length(path_list, root, distance, matrix)

    # init
    group = []
    for _ in select_range:
        group.append((select_len, copy_list(select_path_list)))

    change_cnt = 0
    outer_path_list = select_path_list
    outer_length = select_len

    while time.time() - time_start < time_limit - 6:
        # round_start = time.time()

        next_group = []
        if change_cnt >= static_bare:
            # print("ms trigger")
            change_cnt = 0
            for i in select_range:
                ms_list, ms_len = merge_spilt(n, root, group[i][1], distance, matrix, demand, cap)
                next_group.append((ms_len, ms_list))
            group = next_group
            continue

        for i in group_range:
            new_list, total_len = annealing(n, root, copy_list(group[i % select_size][1]), distance, matrix, local_times, demand, cap)
            next_group.append((total_len, new_list))
        next_group.sort()
        group = next_group[0:select_size]
        select_path_list = group[0][1]
        if select_len > group[0][0]:
            change_cnt = 0
        else:
            change_cnt += 1
        select_len = group[0][0]

        if select_len < outer_length:
            outer_path_list = select_path_list
            outer_length = select_len

        # print("round select len: " + str(select_len) + "   outer len: " + str(outer_length))
        # print(time.time() - round_start)

    return outer_path_list, outer_length


def annealing(n, root, path_list, distance, matrix, times, demand, cap):
    select_path_list = path_list
    select_len = cal_total_length(path_list, root, distance, matrix)

    for i in range(times):
        new_list = vary(select_path_list)
        if (new_list is not None) and legal(new_list, demand, cap):
            new_len = cal_total_length(new_list, root, distance, matrix)
            if new_len < select_len:
                select_path_list = new_list
                select_len = new_len
                # print("update with length" + str(select_len))

    return select_path_list, select_len


def merge_spilt(n, root, path_list, distance, matrix, demand, cap):
    ms_select = random.sample(range(len(path_list)), random.choice(range(len(path_list))) + 1)
    clean_list = []
    ms_path_list = []
    for i in range(len(path_list)):
        if i in ms_select:
            for edge in path_list[i][0]:
                clean_list.append(edge)
        else:
            ms_path_list.append(path_list[i])

    while len(clean_list) != 0:
        path = []
        path_len = 0
        cur = root
        cur_cap = cap
        while True:
            start, edge, index = ms_find_rand(cur, cur_cap, clean_list, distance, demand, root, cur_cap > cap / 2)
            if edge is None:
                break
            path.append(edge)
            path_len += distance[cur][edge[0]] + matrix[edge[0]][edge[1]]
            cur = edge[1]
            cur_cap -= demand[edge[0]][edge[1]]
            clean_list.pop(index)
        path_len += distance[cur][root]
        ms_path_list.append((path, path_len))

    return ms_path_list, cal_total_length(ms_path_list, root, distance, matrix)


def ms_find(cur, cur_cap, clean_list, distance, demand, root, cap_des):
    start = -1
    select_index = -1
    select_edge = None
    min_cost = np.inf
    for i in range(len(clean_list)):
        edge = clean_list[i]
        if cur_cap >= demand[edge[0]][edge[1]]:
            this_min, this_start = (distance[cur][edge[0]], edge[0]) \
                if distance[cur][edge[0]] < distance[cur][edge[1]] \
                else (distance[cur][edge[1]], edge[1])
            if this_min < min_cost:
                start, select_index, select_edge, min_cost = this_start, i, edge, this_min
            elif this_min == min_cost:
                if cap_des and distance[root][this_start] > distance[root][start]:
                    start, select_index, select_edge, min_cost = this_start, i, edge, this_min
                elif (not cap_des) and distance[root][this_start] < distance[root][start]:
                    start, select_index, select_edge, min_cost = this_start, i, edge, this_min
    return start, select_edge, select_index


def ms_find_rand(cur, cur_cap, clean_list, distance, demand, root, cap_des):
    start = -1
    select_index = -1
    select_edge_list = []
    min_cost = np.inf
    for i in range(len(clean_list)):
        edge = clean_list[i]
        if cur_cap >= demand[edge[0]][edge[1]]:
            this_min, this_start = (distance[cur][edge[0]], edge[0]) \
                if distance[cur][edge[0]] < distance[cur][edge[1]] \
                else (distance[cur][edge[1]], edge[1])
            if this_min < min_cost:
                select_edge_list = [(edge, i)]
                min_cost = this_min
            elif this_min == min_cost:
                select_edge_list.append((edge, i))
    if len(select_edge_list) == 0:
        return start, None, select_index
    else:
        edge, index = random.choice(select_edge_list)
        return edge[0], edge, index




if __name__ == "__main__":
    main()
