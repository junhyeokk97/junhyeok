# import p31_sample
from p31_sample import test

x = 222

def main_func():
    print('x: ', x)
main_func()

print("===================")
test()

# x:  222
# ===================
# x:  111       p31_sample.py에 있는 test함수를 불러와서 별도 호출 없이 test()만으로도 사용 가능.
print("===================")
import p31_sample
p31_sample.test()
# 위 test()와 같다.