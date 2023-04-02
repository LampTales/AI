import numpy
import numpy as np
import random
from Project01.proj import AI, directions

LOAD_SWITCH = True
LOOP_TIME = 5
VARY_RATE = 0.4
VARY_STEP = 15
list_size = 20
hold_size = 10

board_list = []

pre_board = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1]])


class BoardStore(object):
    def __init__(self, board):
        self.board = board
        self.play_cnt = 0
        self.win_cnt = 0

    def reset(self, board):
        self.board = board
        self.play_cnt = 0
        self.win_cnt = 0


def win_rate(bs):
    return bs.win_cnt / bs.play_cnt


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
    file.write("it wins with winning rate " + str(win_rate(board)) + "\n")
    file.write(str(board.board) + "\n\n")
    file.close()


def mix(b1, b2):
    board = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if random.random() <= 0.5:
                board[i][j] = b1[i][j]
            else:
                board[i][j] = b2[i][j]
    return board


def vary(board, rate=VARY_RATE, step=VARY_STEP):
    for i in range(8):
        for j in range(8):
            if random.random() < rate:
                board[i][j] += random.random() * step - step / 2
    return board


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
        white_own.win_cnt += 1
    elif win_flag == -1:
        black_own.win_cnt += 1




def main():
    board = np.array(pre_board)
    if LOAD_SWITCH:
        board = load_board()
    board_list.append(BoardStore(board))
    for i in range(list_size - 1):
        nb = np.array(board)
        board_list.append(BoardStore(vary(nb, rate=0.75)))

    for rcnt in range(LOOP_TIME):
        host_board = board_list[0]
        print("loop " + str(rcnt + 1) + " began")
        # compete
        for i in range(list_size):
            if i == list_size - 1:
                break
            for j in range(i + 1, list_size):
                print("race between " + str(i) + " and " + str(j) + " began")
                white_player = AI(8, 1, 5, start_dep=4, score_board_1=board_list[i].board)
                black_player = AI(8, -1, 5, start_dep=4, score_board_1=board_list[j].board)
                compete(white_player, black_player, board_list[i], board_list[j])
                white_player = AI(8, 1, 5, start_dep=4, score_board_1=board_list[j].board)
                black_player = AI(8, -1, 5, start_dep=4, score_board_1=board_list[i].board)
                compete(white_player, black_player, board_list[j], board_list[i])
                print("race between " + str(i) + " and " + str(j) + " finished")

        # save information
        board_list.sort(key=win_rate, reverse=True)
        for b in board_list:
            print(win_rate(b))
        save_board(board_list[0].board)
        save_information(board_list[0], rcnt + 1, host_board == board_list[0])
        print("loop " + str(rcnt + 1) + " finished")
        ncnt = hold_size
        for i in range(5):
            for j in range(i + 1, 5):
                board_list[ncnt] = vary(mix(board_list[i].board, board_list[j].board))
                ncnt += 1
                if ncnt == list_size:
                    break
            if ncnt == list_size:
                break



if __name__ == "__main__":
    main()
