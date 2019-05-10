import math
import numpy as np
import tensorflow as tf

def count_down(number):
    print("start")
    while number>0:

        number-=1
        yield number + 10

res = count_down(3)
while 1:
    c = next(res)
    if c is None:
        break
    print(c)
    print("******************************")

print("************************************************************")
print("************************************************************")

