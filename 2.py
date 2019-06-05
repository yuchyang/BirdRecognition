import random
k = 2
list=[]
a = 1
b = 100
isOdd = True
while len(list) < 2:
    i = random.randint(a, b)
    if i not in list:
        for k in range(2,i):
            if i%k == 0:
                flag = False
        if flag == True:
            list.append(i)
        else:
            flag = True
print(list)