import time
now = time.time()
print()
for x in range(1, 6):
    print(x)
i = 0
time.sleep(5)
print(now - time.time())