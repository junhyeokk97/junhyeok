param_bouns = {'x1' :(-1,5),
               'x2' : (0,4)}

def y_function(x1,x2):
    return -x1 **2 - (x2 -2) **2 + 10

# pip install bayesian-optimization

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds = param_bouns,
    random_state = 333 
) 

optimizer.maximize(init_points = 15,
                   n_iter = 30)

print(optimizer.max)