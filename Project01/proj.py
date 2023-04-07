import numpy as np
import random
import queue
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
MODE = 0
directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
half_dirs = [(1, 0), (0, 1), (1, -1), (1, 1)]
static_score_pre = -3
static_score = -15
avai_score = -20
chess_socre = -10
STATIC_DEPTH = 32
SWITCH_DEPTH = 5

sample = np.array([[-100, 70, -40, -10, -10, -40, 50, -100],
                   [70, 0, -1, -1, -1, -1, 0, 50],
                   [-40, -1, -3, -2, -2, -3, -1, -40],
                   [-10, -1, -2, -1, -1, -2, -1, -10],
                   [-10, -1, -2, -1, -1, -2, -1, -10],
                   [-40, -1, -3, -2, -2, -3, -1, -40],
                   [50, 0, -1, -1, -1, -1, 0, 50],
                   [-100, 50, -40, -10, -10, -40, 50, -100]])


random.seed(0)


# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out, start_dep=3, score_board_1=sample):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        # my finish counter
        self.REST = 64
        self.DEPTH = start_dep
        self.begin_time = 0
        self.score_board_1 = score_board_1
        self.dep = self.DEPTH

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.find_legal(chessboard)
        if len(self.candidate_list) == 0:
            return
        self.dep = self.DEPTH
        while True:
            _, spot, terminate = self.cal_max(chessboard, -1000000, 1000000, self.dep)
            if terminate or self.dep > self.REST:
                # print("reach dep ", self.dep - 1)
                break
            else:
                if spot != (-1, -1):
                    self.candidate_list.append(spot)
                self.dep += 1



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
            return self.get_score(chessboard), (-1, -1), False

        total_max = -1000000
        total_spot = spots[0]
        for spot in spots:
            board = self.flip_over(chessboard, spot, self.color)
            cur_val, _, _ = self.cal_min(board, alpha, beta, step_cnt - 1)
            if cur_val > total_max:
                total_max, total_spot = cur_val, spot
            if total_max >= beta:
                break
            if total_max > alpha:
                alpha = total_max
            if time.time() - self.begin_time > self.time_out - 0.1:
                return total_max, total_spot, True
        return total_max, total_spot, False

    def cal_min(self, chessboard, alpha, beta, step_cnt):
        spots = self.get_spots(chessboard, -self.color)
        if step_cnt == 0 or len(spots) == 0:
            return self.get_score(chessboard), (-1, -1), False

        total_min = 1000000
        total_spot = spots[0]
        for spot in spots:
            board = self.flip_over(chessboard, spot, -self.color)
            cur_val, _, _ = self.cal_max(board, alpha, beta, step_cnt - 1)
            if cur_val < total_min:
                total_min, total_spot = cur_val, spot
            if total_min <= alpha:
                break
            if total_min < beta:
                beta = total_min
            if time.time() - self.begin_time > self.time_out - 0.1:
                return total_min, total_spot, True
        return total_min, total_spot, False

    def get_score(self, chessboard):
        if self.REST > self.dep:
            brd = 0
            brd = self.color * np.sum(chessboard * self.score_board_1)

            chs = 0
            chs = self.color * np.sum(chessboard) * chess_socre

            avi = 0
            if self.dep % 2 == 0:
                avi = -len(self.get_spots(chessboard, self.color))
            else:
                avi = len(self.get_spots(chessboard, -self.color))

            sta = 0
            sta = self.get_static(chessboard)

            return brd + chs + avi * avai_score + sta * static_score
            # return  avi * avai_score + sta * static_score
        else:
            return -self.color * np.sum(chessboard)

    # the early version of finding static spots, probably a piece of ****
    def get_static_pre(self, chessboard):
        static_board = np.zeros((8, 8))
        # if chessboard[0][0] == 0 and chessboard[0][7] == 0 and chessboard[7][0] == 0 and chessboard[7][7] == 0:
        #     return 0

        for i in range(8):
            for j in range(8):
                c = chessboard[i][j]
                if c == 0:
                    continue
                for dx, dy in directions:
                    x = i + dx
                    y = j + dy
                    flag = True
                    while 0 <= x < 8 and 0 <= y < 8:
                        if chessboard[x][y] != c:
                            if chessboard[x][y] == 0:
                                flag = False
                            break
                        x = x + dx
                        y = y + dy
                    if flag:
                        static_board[i][j] += c
        return self.color * np.sum(static_board)

    def get_static(self, chessboard):
        static_board = np.zeros((8, 8))
        if chessboard[0][0] == 0 and chessboard[0][7] == 0 and chessboard[7][0] == 0 and chessboard[7][7] == 0:
            return 0

        if chessboard[0][0] != 0:
            c = chessboard[0][0]
            static_board[0][0] = c
            for x in range(1, 7):
                if c == chessboard[0][x]:
                    static_board[0][x] = c
                else:
                    break
            for x in range(1, 7):
                if c == chessboard[x][0]:
                    static_board[x][0] = c
                else:
                    break

        if chessboard[0][7] != 0:
            c = chessboard[0][7]
            static_board[0][7] = c
            for x in range(1, 7):
                if c == chessboard[x][7]:
                    static_board[x][7] = c
                else:
                    break
            for x in range(1, 7):
                if c == chessboard[0][7 - x]:
                    static_board[0][7 - x] = c
                else:
                    break

        if chessboard[7][0] != 0:
            c = chessboard[7][0]
            static_board[7][0] = c
            for x in range(1, 7):
                if c == chessboard[7][x]:
                    static_board[7][x] = c
                else:
                    break
            for x in range(1, 7):
                if c == chessboard[7 - x][0]:
                    static_board[7 - x][0] = c
                else:
                    break

        if chessboard[7][7] != 0:
            c = chessboard[7][7]
            static_board[7][7] = c
            for x in range(1, 7):
                if c == chessboard[7][7 - x]:
                    static_board[7][7 - x] = c
                else:
                    break
            for x in range(1, 7):
                if c == chessboard[7 - x][7]:
                    static_board[7 - x][7] = c
                else:
                    break

        for i in range(1, 7):
            for j in range(1, 7):
                c = chessboard[i][j]
                if c == 0:
                    continue
                non_sta = False
                for dx, dy in half_dirs:
                    x = i + dx
                    y = j + dy
                    flag = False
                    opposite = 0
                    while 0 <= x < 8 and 0 <= y < 8:
                        if chessboard[x][y] != c:
                            opposite = chessboard[x][y]
                            flag = True
                            break
                        x = x + dx
                        y = y + dy
                    if flag:
                        x = i - dx
                        y = j - dy
                        while 0 <= x < 8 and 0 <= y < 8:
                            if chessboard[x][y] != c:
                                if chessboard[x][y] == opposite and opposite == -c:
                                    break
                                non_sta = True
                                break
                            x = x - dx
                            y = y - dy
                    if non_sta:
                        break
                if not non_sta:
                    static_board[i][j] = c

        return self.color * np.sum(static_board)

