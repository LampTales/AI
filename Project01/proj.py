import numpy as np
import random
import queue
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
static_score = -10
score_table = np.array([[1, 8, 3, 7, 7, 3, 8, 1],
                        [8, 3, 2, 5, 5, 2, 3, 8],
                        [3, 2, 6, 6, 6, 6, 2, 3],
                        [7, 5, 6, 4, 4, 6, 5, 7],
                        [7, 5, 6, 4, 4, 6, 5, 7],
                        [3, 2, 6, 6, 6, 6, 2, 3],
                        [8, 3, 2, 5, 5, 2, 3, 8],
                        [1, 8, 3, 7, 7, 3, 8, 1]])

modify_table = np.array([[-100, -10, -20, -5, -5, -20, -10, -100],
                         [-10, -10, -8, -5, -5, -8, -10, -10],
                         [-20, -8, -4, -4, -4, -4, -8, -20],
                         [-5, -5, -4, -6, -6, -5, -5, -5],
                         [-5, -5, -4, -6, -6, -5, -5, -5],
                         [-20, -8, -4, -4, -4, -4, -8, -20],
                         [-10, -10, -8, -5, -5, -8, -10, -10],
                         [-100, -10, -20, -5, -5, -20, -10, -100]])

pos_table = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1]])

sample = np.array([[-500, 25, -20, -10, -10, -20, 25, -500],
                   [25, 45, -1, -1, -1, -1, 45, 25],
                   [-20, -1, -3, -2, -2, -3, -1, -20],
                   [-10, -1, -2, -1, -1, -2, -1, -10],
                   [-10, -1, -2, -1, -1, -2, -1, -10],
                   [-20, -1, -3, -2, -2, -3, -1, -20],
                   [25, 45, -1, -1, -1, -1, 45, 25],
                   [-500, 25, -20, -10, -10, -20, 25, -500]])

SWITCH_DEPTH = 5
random.seed(0)


# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out, dep=5, score_board_1=sample, score_board_2=pos_table):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        # my finish counter
        self.REST = 64
        self.DEPTH = dep
        self.begin_time = 0
        self.score_board_1 = score_board_1
        self.score_board_2 = score_board_2

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.find_legal(chessboard)
        _, spot = self.cal_max(chessboard, -10000, 10000, self.DEPTH)
        if spot != (-1, -1):
            self.candidate_list.append(spot)

        # ==============Find new pos========================================
        # Make sure that the position of your decision on the chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chessboard
        # You need to add all the positions which are valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # candidate_list example: [(3,3),(4,4),(4,4)]
        # we will pick the last element of the candidate_list as the position you choose.
        # In above example, we will pick (4,4) as your decision.
        # If there is no valid position, you must return an empty

    def find_legal(self, chessboard):
        self.begin_time = time.time()
        p_list = np.where(chessboard == COLOR_NONE)
        self.REST = len(p_list[0])
        for p in zip(p_list[0], p_list[1]):
            flag = False
            for dx, dy in directions:
                x = p[0] + dx
                y = p[1] + dy
                while 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == -self.color:
                    x = x + dx
                    y = y + dy
                    if 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == self.color:
                        self.candidate_list.append((p[0], p[1]))
                        flag = True
                        break
                if flag:
                    break

    def get_spots(self, chessboard, color):
        p_list = np.where(chessboard == COLOR_NONE)
        spots = []
        for p in zip(p_list[0], p_list[1]):
            flag = False
            for dx, dy in directions:
                x = p[0] + dx
                y = p[1] + dy
                while 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == -color:
                    x = x + dx
                    y = y + dy
                    if 0 <= x < 8 and 0 <= y < 8 and chessboard[x][y] == color:
                        spots.append((p[0], p[1]))
                        flag = True
                        break
                if flag:
                    break
        return spots

    def flip_over(self, chessboard, spot, color):
        board = np.array(chessboard)
        board[spot[0]][spot[1]] = color
        for dx, dy in directions:
            x = spot[0] + dx
            y = spot[1] + dy
            flag = False
            while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == -color:
                x = x + dx
                y = y + dy
                if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == color:
                    flag = True
                    break
            if flag:
                nx = x - dx
                ny = y - dy
                while board[nx][ny] != color:
                    board[nx][ny] = color
                    nx = nx - dx
                    ny = ny - dy
        return board

    def cal_max(self, chessboard, alpha, beta, step_cnt):
        spots = self.get_spots(chessboard, self.color)
        if step_cnt == 0 or len(spots) == 0:
            return self.get_score(chessboard), (-1, -1)

        total_max = -10000
        total_spot = spots[0]
        for spot in spots:
            board = self.flip_over(chessboard, spot, self.color)
            cur_val, _ = self.cal_min(board, alpha, beta, step_cnt - 1)
            if cur_val > total_max:
                total_max, total_spot = cur_val, spot
            if total_max >= beta:
                break
            if total_max > alpha:
                alpha = total_max
            if time.time() - self.begin_time > self.time_out - 0.1:
                break
        return total_max, total_spot

    def cal_min(self, chessboard, alpha, beta, step_cnt):
        spots = self.get_spots(chessboard, -self.color)
        if step_cnt == 0 or len(spots) == 0:
            return self.get_score(chessboard), (-1, -1)

        total_min = 10000
        total_spot = spots[0]
        for spot in spots:
            board = self.flip_over(chessboard, spot, -self.color)
            cur_val, _ = self.cal_max(board, alpha, beta, step_cnt - 1)
            if cur_val < total_min:
                total_min, total_spot = cur_val, spot
            if total_min <= alpha:
                break
            if total_min < beta:
                beta = total_min
            if time.time() - self.begin_time > 4.9:
                break
        return total_min, total_spot

    def get_score(self, chessboard):
        if self.REST > SWITCH_DEPTH:
            return np.sum(self.color * chessboard * self.score_board_1)
        else:
            return np.sum(self.color * chessboard * self.score_board_2)


