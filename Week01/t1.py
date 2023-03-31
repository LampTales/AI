def is_next(t,p,m,h,w):
    return t != p and 0 <= t[0] < h and 0 <= t[1] < w and m[t[0]][t[1]] == '#'


def can_go(t,m,h,w):
    return 0 <= t[0] < h and 0 <= t[1] < w and m[t[0]][t[1]] == '-'


test_case = 4
with open(f'test_cases/{test_case}-map.txt', 'r') as f:
    game_map = [list(line.strip()) for line in f.readlines()]
with open(f'./test_cases/{test_case}-actions.txt', 'r') as f:
    actions = [*map(int, f.read().split(' '))]

headSpot = (0, 0)
x = len(game_map)
y = len(game_map[0])

# loading head
for i in range(x):
    for j in range(y):
        if game_map[i][j] == '@':
            headSpot = (i, j)

# loading body
preSpot = (-1, -1)
nowSpot = headSpot
queue = [headSpot]
while True:
    following = (nowSpot[0] - 1, nowSpot[1])
    if is_next(following,preSpot,game_map,x,y):
        queue.append(following)
        preSpot = nowSpot
        nowSpot = following
        continue
    following = (nowSpot[0] + 1, nowSpot[1])
    if is_next(following,preSpot,game_map,x,y):
        queue.append(following)
        preSpot = nowSpot
        nowSpot = following
        continue
    following = (nowSpot[0], nowSpot[1] - 1)
    if is_next(following,preSpot,game_map,x,y):
        queue.append(following)
        preSpot = nowSpot
        nowSpot = following
        continue
    following = (nowSpot[0], nowSpot[1] + 1)
    if is_next(following,preSpot,game_map,x,y):
        queue.append(following)
        preSpot = nowSpot
        nowSpot = following
        continue
    break

for i in range(len(actions)):
    nextSpot = queue[0]
    if actions[i] == 0:
        nextSpot[0] -= 1
    elif actions[i] == 1:
        nextSpot[0] += 1
    elif actions[i] == 2:
        nextSpot[1] -= 1
    else:
        nextSpot[1] += 1

    if can_go(nextSpot,game_map,x,y):
        queue.insert(0,nextSpot)
        queue.pop()
    else:
        print(i)

print(queue[0])