import random as rm

ss1 = rm.random() * 0.02
ss2 = rm.random() * 0.02
ss1 = round(ss1, 5)
ss2 = round(ss2, 5)

print(ss1, ss2)


print(int(((ss1 + ss2)/0.04)*255))