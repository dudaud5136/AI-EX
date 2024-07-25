# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:06:32 2024

@author: jyshin
"""

import numpy as np

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1
    
    
def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1
    
    
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1
    

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

def FULLADDER(x1,x2,c):
    y1 = XOR(x1,x2)
    y2 = XOR(y1,c)
    
    y3 = AND(c,y1)
    y4 = AND(x2,x1)
    y5 = OR(y3,y4)
    
    return [y2,y5]
    
print("FULL Adder 0 0 0 : {0}".format(FULLADDER(0,0,0)))
print("FULL Adder 0 0 1 : {0}".format(FULLADDER(0,0,1)))
print("FULL Adder 0 1 0 : {0}".format(FULLADDER(0,1,0)))
print("FULL Adder 0 1 1 : {0}".format(FULLADDER(0,1,1)))
print("FULL Adder 1 0 0 : {0}".format(FULLADDER(1,0,0)))
print("FULL Adder 1 0 1 : {0}".format(FULLADDER(1,0,1)))
print("FULL Adder 1 1 0 : {0}".format(FULLADDER(1,1,0)))
print("FULL Adder 1 1 1 : {0}".format(FULLADDER(1,1,1)))

