param_bouns = {'x1': (-1,5),
               'x2': (0,4)}

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 + 10

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,     # 블랙박스 함수.
    pbounds=param_bouns,
    random_state=50,
)

optimizer.maximize(init_points=15,    # init_point >> 최적화 시작 지점 또는 초기 시도 횟수
                   n_iter=30)

print(optimizer.max)