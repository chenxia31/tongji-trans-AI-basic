# New-ton method
# 目的是通过迭代的方式找到 f(x) 的最小值

import random
import math 
def f(x):
    ''' 
    输入 x 得到对应的函数输出值
    '''
    return math.sin(x) + x*x + x*2 - 5

def df(f):
    ''' 
    返回任意函数的梯度
    '''
    def no_name(x):
        return 100*(f(x+0.01)-f(x))
    return no_name

def is_end(x):
    ''' 
    终止条件
    '''
    return abs(x-0) > 1e-15

def newton_update(f,df):
    ''' 
    牛顿法的更新方式
    '''
    def no_name(x):
        # NOTE 这里是填写代码的区域
        
        # END 
        return x  
    return no_name


def init_val():
    return random.randint(0,10)

def base_iterations(f):
    guess = init_val()
    while is_end(f(guess)):
        guess = newton_update(f,df(f))(guess)
        print(guess)
    return guess 

print(base_iterations(f))


