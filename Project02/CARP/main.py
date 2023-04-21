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
    return None





def main():
    args = get_args()
    print(args)
    read_data(args.file_name)


if __name__ == "__main__":
    main()
