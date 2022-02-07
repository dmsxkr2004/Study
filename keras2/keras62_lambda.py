from numpy import gradient

gradient = lambda x: 2*x -4

def gradient2(x): # 위의 식과 같다
    return 2*x - 4

x = 3

print(gradient(x))
print(gradient2(x))
'''
2
2
'''