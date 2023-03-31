import numpy
import numpy as np
import random
from Project01.proj import AI, directions

TRAINING_RATE = 15

list_size = 20
out_size = 10
board_list = []


class BoardStore(object):
    def __init__(self, board):
        self.board = board
        self.play_cnt = 0
        self.win_cnt = 0

    def reset(self, board):
        self.board = board
        self.play_cnt = 0
        self.win_cnt = 0


def save_board(board):
    np.savetxt("board.txt", board)


def load_board():
    return numpy.loadtxt("board.txt")


def save_information(board, loop, change):
    file = open("save.txt", "a")
    file.write("Looping time: " + str(loop) + "\n")
    if change:
        file.write("best board changed\n")
    else:
        file.write("best board unchanged\n")
    file.write(str(board) + "\n\n")
    file.close()


def generate_new_board(board):
    new_board = np.array(board)
    for i in range(8):
        for j in range(8):
            new_board[i][j] += random.random() * TRAINING_RATE - TRAINING_RATE / 2
    return new_board


def judge(chessboard):
    sum = -np.sum(chessboard)
    if sum > 0:
        return 1
    elif sum < 0:
        return -1
    else:
        return 0


def legal_left(chessboard):
    p_list = np.where(chessboard == 0)
    cand_list = []
    for color in [1, -1]:
        for p in zip(p_list[0], p_list[1]):
            flag = False
            for dx, dy in directions:
                x = p[0] + dx
                y = p[1] + dy
                while 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == -color:
                    x = x + dx
                    y = y + dy
                    if 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == color:
                        cand_list.append((p[0], p[1]))
                        flag = True
                        break
                if flag:
                    break
    return len(cand_list) != 0


def compete(white_player, black_player, white_own, black_own):
    white_own.play_cnt += 1
    black_own.play_cnt += 1
    chessboard = np.zeros((8, 8), int)
    chessboard[3][3] = 1
    chessboard[4][4] = 1
    chessboard[3][4] = -1
    chessboard[4][3] = -1
    cur = black_player
    while legal_left(chessboard):
        cur.go(chessboard)
        if len(cur.candidate_list) != 0:
            chessboard = cur.flip_over(chessboard, cur.candidate_list[-1], cur.color)
        if cur == black_player:
            cur = white_player
        else:
            cur = black_player
    win_flag = judge(chessboard)
    if win_flag == 1:
        return 1




def main():
    board = load_board()
    board_list.append(BoardStore(board))
    for i in range(list_size - 1):
        board_list.append(BoardStore(generate_new_board(board)))
    for i in range(list_size):
        for j in range(list_size - 1 - i):
            white_player = AI(8, 1, 5.5, score_board_1=board_list[i].board)
            black_player = AI(8, -1, 5.5, score_board_1=board_list[j].board)



if __name__ == "__main__":
    main()
