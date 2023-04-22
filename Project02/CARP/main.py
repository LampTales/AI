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
    file.readline()
    header = dict()
    for _ in range(7):
        line = file.readline()
        data = line.split(':')
        header[data[0].strip().lower()] = int(data[1].strip())
    print(header)
    file.readline()
    line = file.readline()
    while line.strip() != 'END':
        data = line.split()
        line = file.readline()

class Node(object):
    def __init__(self, node_id):
        self.node_id = node_id
        self.edge_list = []





def main():
    args = get_args()
    print(args)
    read_data(args.file_path)


if __name__ == "__main__":
    main()
