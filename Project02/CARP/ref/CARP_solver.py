import argparse
from read_dat import readData
from Problem import Problem
from naive_solver import Solver

def read_from_command():
    parser = argparse.ArgumentParser(description="Solve the CARP problem")
    parser.add_argument("input_file", type=argparse.FileType('r'), help="Input file path")
    parser.add_argument("-t", "--termination", type=int, default=60, help="Termination time")
    parser.add_argument("-s", "--randomseed", type=int, help="Random seed")
    args = parser.parse_args()
    input_file_name = args.input_file.name
    termination = args.termination
    randomSeed = args.randomseed
    print("----This is the read_from_command function of CARP_solver.py ----")
    print("Input file name: {}".format(input_file_name))
    print("Termination condition: {}".format(termination))
    print("Random seed: {}".format(randomSeed))
    print("---------------------------------------------------------------")
    return input_file_name, termination, randomSeed

if __name__ == "__main__":
    input_file_contents, termination, randomSeed = read_from_command()
    dataDict, route = readData(input_file_contents)
    problem = Problem(dataDict, route)
    solver = Solver(problem, termination, randomSeed)
    solver.solve()

