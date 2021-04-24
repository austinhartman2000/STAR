# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:04:59 2020

@author: Austin Hartman
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

player = "Meyers Leonard"

def toPercentiles(zscores):
    newscores = []
    for score in zscores:
        newscores.append(scipy.stats.norm.cdf(score))
    return newscores

data = pd.read_csv("archetypes3.csv")

# Example data
plt.rcdefaults()
fig, ax = plt.subplots()

skills = ('Offensive Talent', 'Interior Focus', 'Inefficiency', 'Perimeter Defense', 'Unicorn')
y_pos = np.arange(len(skills))
performance = data[(data['Player'] == player) & (data['Year'] == 2019)]
performance = performance[['A', 'B', 'C', 'D', 'E']].values[0]
print(performance)
performance = toPercentiles(performance)

ax.barh(y_pos, performance, align='center', color = ['red', 'deepskyblue', 'orange', 'lawngreen', 'mediumorchid'])
ax.set_yticks(y_pos)
ax.set_yticklabels(skills)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Percentile')
ax.set_title(player + ' Player Profile')

plt.show()