lst = [1,2,3]
nums = iter(lst)

# print(nums.next())   # 파이썬 2.0 문법
print(next(nums))   # 1
print(next(nums))   # 2
print(next(nums))   # 3
print(next(nums))   # error // 반복되는 함수에선 처음으로 되돌아간다.

