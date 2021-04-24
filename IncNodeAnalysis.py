# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 02:24:13 2020

@author: Austin Hartman
"""
import matplotlib.pyplot as plt

x = range(2,11)

a = [104.902004, 32.991321, 18.795906, 19.129985, 21.089647, 16.252083, 19.744626, 15.950799, 17.833067]
b = [15.889632, 16.858192, 18.316267, 17.582414, 18.567074, 17.302182, 15.376222, 17.338434, 17.321349]
c = [25.789373, 30.256018, 42.345212, 29.761203, 26.914591, 59.659127, 31.306611, 43.567166, 56.336652]
d = [20.192517, 18.452193, 19.572094, 21.602540, 22.320396, 20.055514, 19.355803, 18.227373, 25.463807]
e = [18.624255, 19.638950, 20.567170, 24.385507, 17.613486, 21.490404, 16.593928, 16.499121, 21.106735]

plt.plot(x,a, label = "Offensive Talent", color = 'red')
plt.plot(x,b, label = "Interior Focus", color = 'deepskyblue')
plt.plot(x,c, label = "Inefficiency", color = 'orange')
plt.plot(x,d, label = "Perimeter Defense", color = 'lawngreen')
plt.plot(x,e, label = "Unicorn", color = 'mediumorchid')
plt.xlabel("Player VORP Ranking within team")
plt.ylabel("Inc Node Purity")
plt.title("Inc Node Purity of Five Factors for each player")
plt.legend()
plt.show()