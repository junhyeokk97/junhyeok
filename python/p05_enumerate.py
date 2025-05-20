list = ['a', 'b', 'c', 'd', 5]
print(list)

for value in list:
   print(value)
    # a
    # b
    # c
    # d
    # 5
for index, value in enumerate(list):        # enumerate는 key(index), value 값을 나열해준다.
    print(index, value)
    # 0 a
    # 1 b
    # 2 c
    # 3 d
    # 4 5
    
