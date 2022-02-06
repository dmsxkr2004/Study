from bayes_opt import BayesianOptimization

def black_box_function(x, y):
    return -x **2 - (y - 1) ** 2 + 1

pbounds = {'x' : (-3, 4), 'y' : (-6, 3)}

optimizer = BayesianOptimization(
    f = black_box_function, # f 모델이 들어간다.
    pbounds = pbounds, # 파라미터가 들어간다.
    random_state=66
)

optimizer.maximize(
    init_points = 2,
    n_iter = 15
)
