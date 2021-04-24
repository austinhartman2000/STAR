# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:24:13 2020

@author: Austin Hartman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:59:06 2020

@author: Austin Hartman
"""

import pandas as pd
import re

def fixNames(name):
    newname = re.sub("[\*]*\\\\.+", "", name)
    return newname

data = pd.read_csv("modelFullData.csv")
data['Player'] = data['Player'].apply(fixNames)
data = data.fillna(0)
teams = []
for team in data['Tm'].values:
    if team not in teams:
        teams.append(team)
teams.remove('TOT')
years = []
for i in range(1985,2020):
    years.append(i)
teamPlayers = []
teamNames = []
for team in teams:
    for year in years:
        newdata = data[(data['Tm'] == team) & (data['Year'] == year) & (data['MP'] >= 50)]
        if(len(newdata)>0):
            newdata = newdata.sort_values(by=['MP'], ascending = False)
            newdata = newdata[:10]
            newdata = newdata.sort_values(by = ['VORP'], ascending = False)
            names = []
            for i in range(0,10):
                names.append(newdata['Player'].values[i])
            teamPlayers.append(names)
            teamNames.append(team + " " + str(year))
#print(teamNames)
#print(len(teamNames))
#print(len(teamPlayers))

archetypes = pd.read_csv("archetypes3.csv")
srs = pd.read_csv("fullModelSRS.csv")
teamVectors = []

archetypes = archetypes[archetypes['Tm'] != 'TOT']

for i in range(0,len(teamNames)):
        teamVectors.append([teamNames[i][0:3],teamNames[i][4:], 0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0])
print(teamPlayers[0])
for i in range(0,len(teamVectors)):
    team = teamPlayers[i]
    place = 2
    for j in range(len(team)):
        player = team[j]
        #print(teamVectors[i][0], int(teamVectors[i][1]), player)
        teamVectors[i][place] = archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][0]
        place+=1
        teamVectors[i][place] = archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][4]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][5]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][6]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][7]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][8]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][9]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][10]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][11]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][12]
        place+=1
        teamVectors[i][place] += archetypes[(archetypes['Player'] == player) & (archetypes['Year'] == int(teamVectors[i][1]))].values[0][13]
        place+=1
    teamVectors[i][112] = srs[(srs['Tm'] == teamVectors[i][0]) & (srs['Season'] == int(teamVectors[i][1]))].values[0][10]
        

frame = pd.DataFrame(teamVectors)
cols = ['Team', 'Year']
for i in range(10):
    cols.append("Name" + str(i))
    cols.append("Arch" + str(i))
    cols.append("A" + str(i))
    cols.append("B" + str(i))
    cols.append("C" + str(i))
    cols.append("D" + str(i))
    cols.append("E" + str(i))
    cols.append("OBPM" + str(i))
    cols.append("DBPM" + str(i))
    cols.append("VORP" + str(i))
    cols.append('MP' + str(i))
cols.append('SRS')
frame.columns = cols
frame.to_csv(r'C:\Users\Austin Hartman\Documents\The Model Folder\modelFinalDataCategories.csv', index = False)