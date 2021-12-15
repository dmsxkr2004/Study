import numpy as np

a = np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)

x = bbb[:, :4]
y = bbb[:, 4]
print(x,y)
print(x.shape, y.shape) # (6, 4) (6,)