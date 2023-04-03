import numpy as np
from proj import AI
from Developping.main import judge, legal_left

chessboard = np.zeros((8, 8), int)
print(chessboard)
chessboard[3][3] = 1
chessboard[4][4] = 1
chessboard[3][4] = -1
chessboard[4][3] = -1
print(chessboard)
p1 = AI(8, 1, 5, )
p2 = AI(8, -1, 5, )
cur = p2
while legal_left(chessboard):
    cur.go(chessboard)
    if len(cur.candidate_list) != 0:
        chessboard = cur.flip_over(chessboard, cur.candidate_list[-1], cur.color)
    else:
        if cur == p1:
            cur = p2
        else:
            cur = p1
        continue
    print("player color: " + str(cur.color) + " , choose spot: " + str(cur.candidate_list[-1]) + " , and get: " + str(cur.get_score(chessboard)))
    print(chessboard)
    if cur == p1:
        cur = p2
    else:
        cur = p1


print("the winner is " + str(judge(chessboard)))