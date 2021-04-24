# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:09:18 2020

@author: Austin Hartman
"""

import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import re
import scipy
import statistics as s
import seaborn as sns
import pylab
import unicodedata

def convertArch(point):
    point = ord('A') + point
    point = chr(point)
    return point

def fixNames(name):
    newname = re.sub("[\*]*\\\\.+", "", name)
    newname = unicodedata.normalize('NFKD', newname).encode('ASCII', 'ignore')
    return (newname).decode("utf-8")

def toPercentiles(zscores):
    newscores = []
    for score in zscores:
        newscores.append(scipy.stats.norm.cdf(score))
    return newscores

data = pd.read_csv("modelFullData.csv")
data = data.fillna(0)
print(data.loc[13])

data = data[(data['MP'] >= 100) & (data['Tm'] != 'TOT')]
data = data.reset_index(drop=True)

tscale = data.drop(columns = ["Player","Rk", "Pos", "Tm", "Year"])

tscale = preprocessing.scale(tscale, with_mean = True, with_std = True)

pca = decomposition.PCA(n_components=2)
pca.fit(tscale)
X_trans = pca.transform(tscale)

plt.scatter(X_trans[:,0], X_trans[:,1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Two component decompostion")
plt.show()

pca = decomposition.PCA(n_components=3)
pca.fit(tscale)
X_trans = pca.transform(tscale)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_trans[:,0], X_trans[:,1], X_trans[:,2])
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.set_title("Three component decompostion")
plt.show()

explained_var = pca.explained_variance_ratio_
print(explained_var)
print(sum(explained_var))

k = 1
while(sum(explained_var) <= 0.95):
    k = k+1
    pca2 = decomposition.PCA(n_components = k)
    pca2.fit(tscale)
    X_trans2 = pca2.transform(tscale)
    explained_var = pca2.explained_variance_ratio_
    print(k)
    print(sum(explained_var))


#=============================================================================
#kValues = range(2,15)
#inertias = []
#silScores = []
#for kValue in kValues:
#    kmeans = KMeans(n_clusters = kValue, random_state = 0).fit(tscale)
#    preds = kmeans.labels_
#    inertias.append(kmeans.inertia_)
#    silScores.append(silhouette_score(tscale, preds, metric = 'euclidean'))
# 
#plt.plot(kValues, inertias, marker = "o")
#plt.xlabel("k value")
#plt.ylabel("inertia")
#plt.title("Inertia Values over K Values")
#plt.show()
#
#plt.plot(kValues, silScores, marker = "o")
#plt.xlabel("k value")
#plt.ylabel("Silhouette Score")
#plt.title("Silhouette Scores over K Values")
#plt.show()
# 
#for i in range(13):
#    print("k", i+2, "sil", silScores[i])
#    
#svd = TruncatedSVD(n_components=k)
#svd.fit(tscale)
#plt.plot(range(1,k+1), svd.singular_values_, marker = "o")
#plt.xlabel("Component")
#plt.ylabel("Singular Value")
#plt.title("Singular Values in 32-Component SVD")
#plt.show()
# 
#summedVars = []
#kValues = range(1,k+1)
#for kValue in kValues:
#    summedVars.append(sum(svd.explained_variance_ratio_[0:kValue]))
#for i in range(len(summedVars)):
#    print("k", i+1, "Summed Var", summedVars[i])
#plt.plot(kValues, summedVars, marker = "o")
#plt.xlabel("k-value")
#plt.ylabel("Explained variance")
#plt.title("Total explained variance over different K Values in SVD")
#plt.show()
#=============================================================================

clusters = 7

svd = TruncatedSVD(n_components = 2)
svd.fit(tscale)
svdTransformed = svd.transform(tscale)
kmeans = KMeans(n_clusters = clusters, random_state=0).fit(svdTransformed)
y_kmeans = kmeans.predict(svdTransformed)
centers = np.array(kmeans.cluster_centers_)
plt.scatter(svdTransformed[:,0], svdTransformed[:,1], c=y_kmeans)
plt.scatter(centers[:,0], centers[:,1], marker = 'x', color = 'r')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("K-Means Clustering with k = 7 on 2-Component SVD Data")
plt.show()

svd = TruncatedSVD(n_components = 3)
svd.fit(tscale)
svdTransformed = svd.transform(tscale)
kmeans = KMeans(n_clusters = clusters, random_state=0).fit(svdTransformed)
y_kmeans = kmeans.predict(svdTransformed)
centers = np.array(kmeans.cluster_centers_)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(svdTransformed[:,0], svdTransformed[:,1], svdTransformed[:,2], c=y_kmeans)
ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker = 'x', color = 'r')
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.set_title("K-Means Clustering with k = 7 on 3-Component SVD Data")
plt.show()

svd = TruncatedSVD(n_components = 5)
svd.fit(tscale)
svdTransformed = svd.transform(tscale)
factors = pd.DataFrame(svdTransformed)
factors.columns = ['A', 'B', 'C', 'D', 'E']
print(factors[15495:15500])


print(len(data['Player']))
print(len(factors['A']))

data['A'] = factors['A']
data['B'] = factors['B']
data['C'] = factors['C']
data['D'] = factors['D']
data['E'] = factors['E']

svd = TruncatedSVD(n_components = k)
svd.fit(tscale)
svdTransformed = svd.transform(tscale)
kmeans = KMeans(n_clusters = clusters, random_state=100).fit(svdTransformed)
svdTransformed = svdTransformed.tolist()
centers = np.array(kmeans.cluster_centers_).tolist()
y_kmeans = kmeans.predict(svdTransformed)
data['Archetype'] = y_kmeans
preds = y_kmeans.tolist()
centroid_dist = []
for i in range(len(svdTransformed)):
    centroid_dist.append(scipy.spatial.distance.euclidean(svdTransformed[i], centers[preds[i]]))
dist = np.array(centroid_dist)

pd.set_option("display.max_rows", None)

data['Archetype'] = data['Archetype'].apply(convertArch)
data['Dist'] = dist

data['Player'] = data['Player'].apply(fixNames)

#data[['Player', 'Tm', 'Year', 'Pos', 'Archetype','Dist', 'A', 'B', 'C', 'D', 'E', 'OBPM', 'DBPM', 'VORP', 'MP']].to_csv(r'C:\Users\Austin Hartman\Documents\The Model Folder\archetypes3.csv', index = False)

print(data.iloc[0,])
archs = ["A", "B", "C", "D", "E", "F", "G"]

plt.plot(data['A'].tolist(), data['PTS'].tolist())
plt.xlabel("Factor A")
plt.ylabel("PTS")
plt.title("Factor A vs Points")
plt.show()

plt.plot(data['D'].tolist(), data['PER'].tolist())
plt.xlabel("Factor D")
plt.ylabel("PER")
plt.title("Factor D vs PER")
plt.show()

noCategorical = data.drop(columns = ["Player", "Pos", "Tm", "Archetype", "Dist"])
corr = noCategorical.corr().iloc[-5:,]
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
xticks = np.arange(0,len(noCategorical.columns),1)
yticks = np.arange(0,5,1)
ax.set_xticks(xticks)
plt.xticks(rotation=90)
ax.set_yticks(yticks)
ax.set_xticklabels(noCategorical.columns)
ax.set_yticklabels(noCategorical.columns[-5:])
plt.show()

corr.index.name = 'Factor'
corr.reset_index(inplace=True)
corr.to_csv(r'C:\Users\Austin Hartman\Documents\The Model Folder\FiveFactorsCorrelations.csv', index = False)

factors = ['A', 'B', 'C', 'D', 'E']
for arch in factors:
    print(s.mean(data[arch]), s.stdev(data[arch]))
    data[arch] = (data[arch]-s.mean(data[arch]))/s.stdev(data[arch])
    data[arch+"_std"] = (data[arch]-s.mean(data[arch]))/s.stdev(data[arch])
    print(s.mean(data[arch]), s.stdev(data[arch]))
    print()
    
data = data.rename(columns = {"A_std" : "Offensive Talent", "B_std" : "Interior Focus", "C_std" : "Inefficiency", "D_std" : "Perimeter Defense", "E_std" : "Unicorn"})
data = data.sort_values(by = ['Archetype'])

dataSplits = []
dataCounts = []
for arch in archs:
    newdata = (data[data['Archetype'] == arch])
    print("Arch: " + arch + " count " + str(len(newdata)))
    dataSplits.append(newdata)
    dataCounts.append(len(newdata))

meltedData = pd.melt(dataSplits[0],id_vars=['Archetype'], value_vars = ['Offensive Talent', 'Interior Focus', 'Inefficiency', 'Perimeter Defense', 'Unicorn'], var_name = 'Standardized Factor', value_name = 'Factor Z-Value')
plt.figure(figsize=(15,10))
box = sns.boxplot(x="Archetype", y = 'Factor Z-Value', hue = "Standardized Factor", data= meltedData, palette= "Set1")
plt.axhline(0, ls= '--')
plt.show()


quartiles = np.percentile(dataSplits[0]['Offensive Talent'], [25,50,75])
print(quartiles, toPercentiles(quartiles))
quartiles = np.percentile(dataSplits[0]['Interior Focus'], [25,50,75])
print(quartiles, toPercentiles(quartiles))
quartiles = np.percentile(dataSplits[0]['Inefficiency'], [25,50,75])
print(quartiles, toPercentiles(quartiles))
quartiles = np.percentile(dataSplits[0]['Perimeter Defense'], [25,50,75])
print(quartiles, toPercentiles(quartiles))
quartiles = np.percentile(dataSplits[0]['Unicorn'], [25,50,75])
print(quartiles, toPercentiles(quartiles))

scipy.stats.probplot(data['A'], dist = "norm", plot = pylab)
pylab.show()
scipy.stats.probplot(data['B'], dist = "norm", plot = pylab)
pylab.show()
scipy.stats.probplot(data['C'], dist = "norm", plot = pylab)
pylab.show()
scipy.stats.probplot(data['D'], dist = "norm", plot = pylab)
pylab.show()
scipy.stats.probplot(data['E'], dist = "norm", plot = pylab)
pylab.show()

fig = plt.figure(figsize=(25,10))
plt.rcParams.update({'font.size': 20})
ax = fig.add_axes([0,0,1,1])
archetypes = ["Skilled Big", "Traditional Center", "Offensive Engine", "Secondary Bench Guard", "Bench Big", "Primary Bench Guard", "Secondary Playmakers"]
ax.bar(archetypes,dataCounts)
plt.xlabel("Archetype")
plt.ylabel("Count")
plt.title("Counts of Archetypes in Dataset")
plt.show()
plt.rcParams.update({'font.size': 10})

meltedData = pd.melt(data,id_vars=['Archetype'], value_vars = ['Offensive Talent', 'Interior Focus', 'Inefficiency', 'Perimeter Defense', 'Unicorn'], var_name = 'Standardized Factor', value_name = 'Factor Z-Value')
plt.figure(figsize=(20,10))
box = sns.boxplot(x="Archetype", y = 'Factor Z-Value', hue = "Standardized Factor", data= meltedData, palette= "Set1")
plt.axhline(0, ls= '--')
plt.show()


# Example data
plt.rcdefaults()
fig, ax = plt.subplots()

skills = ('Offensive Talent', 'Interior Focus', 'Inefficiency', 'Perimeter Defense', 'Unicorn')
y_pos = np.arange(len(skills))
performance = data[data['Player'] == "Kristaps Porzingis"]
performance = performance[['A', 'B', 'C', 'D', 'E']].values.tolist()

ax.barh(y_pos, performance, align='center', color = ['red', 'deepskyblue', 'orange', 'lawngreen', 'mediumorchid'])
ax.set_yticks(y_pos)
ax.set_yticklabels(skills)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Percentile')
ax.set_title('Kristaps Porzingis Player Profile')

plt.show()


data[['Player', 'Tm', 'Year', 'Pos', 'Archetype','Dist', 'A', 'B', 'C', 'D', 'E', 'OBPM', 'DBPM', 'VORP', 'MP']].to_csv(r'C:\Users\Austin Hartman\Documents\The Model Folder\archetypes3.csv', index = False)