# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:02:52 2020

@author: Mert
"""

import csv 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# First of all, it is needed to load to data. 
# Our dataset contains 308 columns including the index number, label and 306 features, with 566 rows.
data = [[0]* 308]*566
index = 0 

#Getting all data into data list.
with open('../data/falldetection_dataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data[index] = row
        index += 1

#Seperating features and labels. 
#If the label is F, it is specified as 1, if the label is NF, it is specified as 0.
features = np.zeros((566,306))
labels = np.zeros(566)
for row in range(566):
    label = data[row][1]
    if label == 'F':
        label = 1
    elif label == 'NF':
        label = 0
    labels[row] = label
    features[row] = data[row][2:308]

#perform PCA to reduce the dimensions to 2.
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(features)

# variance is calculated and noted below. 
variance = pca.explained_variance_ratio_
print('Variance of Principal Component 1: ', variance[0])
print('Variance of Principal Component 2: ', variance[1])

#All the data is located to the pandas data frame.
principalDf = pd.DataFrame(data=principalComponents, 
                           columns=['PC1', 'PC2'])
targetDf = pd.DataFrame(data=labels,
                        columns = ['Target'])
allDf = pd.concat([principalDf, targetDf['Target']], axis = 1)

#Then K-means clustering is performed to seperate data into clusters.
#The data which dimensions are reduced is used for K-means clustering.
from sklearn.cluster import KMeans

allDf = allDf.drop(503)
principalDf = principalDf.drop(503)
targetDf = targetDf.drop(503)

#Data is clustered using different number of clusters (N) and error is calculated.
kmeans_error = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                n_init=10, random_state=0)
    kmeans.fit(principalDf)
    kmeans_error.append(kmeans.inertia_)

plt.plot(range(1,10), kmeans_error)
plt.title('Error of each number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Error')
plt.show()

#Data is finally clustered with N=2 clusters.
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300,
                n_init=10, random_state=0)
kmeans.fit(principalDf)
pred_target = kmeans.fit_predict(principalDf)

label1Df = allDf.loc[allDf['Target'] == 1]
label0Df = allDf.loc[allDf['Target'] == 0]

l1 = plt.scatter(label0Df['PC1'], label0Df['PC2'], c='blue')
l2 = plt.scatter(label1Df['PC1'], label1Df['PC2'], c='cyan')
c = plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means clustering algorithm with k=2 clusters')
plt.legend((l1,l2,c),
            ('NF', 'F', 'Cluster Center'),
            loc = 'upper right',
            ncol=3,
            fontsize=8)
plt.show()

#Spliting dataset into training, test and validation.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(principalDf, targetDf, test_size=0.15)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1764)
accuracy = []
classifiers = []

#Train training dataset with SVM with linear kernel.
from sklearn.svm import SVC

def svm_classifier(kernel, c, gamma='auto'):
    svc = SVC(kernel= kernel, gamma=gamma, C=c)
    svc.fit(x_train, y_train['Target'])
    classifiers.append(svc)
    #Calculating validation accuracy score for SVM.
    val_svm = svc.predict(x_val)
    accuracy_svm = 0
    index=0
    for r in y_val['Target']:
        if val_svm[index] == r:
            accuracy_svm += 1
        index += 1
    accuracy_svm = accuracy_svm / x_val.shape[0]
    return accuracy_svm

hyper_parameters_svm=[['linear', 2, 'auto'], ['linear', 1, 'auto'], ['linear', 3, 'auto'], ['rbf', 1, 'scale'],['rbf', 2, 'scale'], ['rbf', 3, 'scale'],
                      ['rbf', 1, 'auto'],['rbf', 2, 'auto'],['rbf', 3, 'auto']]
for i in range(len(hyper_parameters_svm)):
    accuracy.append(svm_classifier(kernel=hyper_parameters_svm[i][0], c=hyper_parameters_svm[i][1],
                                   gamma=hyper_parameters_svm[i][2]))


#Train training dataset with MLP.
from sklearn.neural_network import MLPClassifier

def mlp_classifier(hidden_layer, learning_rate, max_iter):
    classifier = MLPClassifier(hidden_layer_sizes = hidden_layer, learning_rate_init=learning_rate, max_iter=max_iter)
    classifier.fit(x_train, y_train['Target'])
    classifiers.append(classifier)
    #calculating the validation accuracy score for MLPClassifier 1.
    val_mlp = classifier.predict(x_val)
    accuracy_mlp = 0
    index = 0
    for r in y_val['Target']:
        if val_mlp[index] == r:
            accuracy_mlp += 1
        index += 1
    accuracy_mlp = accuracy_mlp / x_val.shape[0]
    return accuracy_mlp

#Trying MLPClassifier with different hyper parameters and get the accuracy scores.
hyper_parameters = [[20,0.01,500], [50,0.01,500], [20,0.05,500], [10, 0.1, 500], [100, 0.01, 500]]
for i in range(len(hyper_parameters)):
    accuracy.append(mlp_classifier(hidden_layer=hyper_parameters[i][0], learning_rate=hyper_parameters[i][1],
                                   max_iter=hyper_parameters[i][2]))  
#After completing all models, print all accuracy result with corresponding model and parameters.
hps = len(hyper_parameters_svm)
for i in range(len(accuracy)):
    if i < hps:
        print('The accuracy score of SVM Classifier with kernel=', hyper_parameters_svm[i][0], 'C=',
              hyper_parameters_svm[i][1], 'gamma=', hyper_parameters_svm[i][2], ': ', '%.4f'%accuracy[i])
    else:
        print('The accuracy score of MLP Classifier with hl_size=', hyper_parameters[i-hps][0], 'lr=',
              hyper_parameters[i-hps][1], 'max iter=', hyper_parameters[i-hps][2], ': ', '%.4f'%accuracy[i])
print("---------------------------------")
#(Cross Validation Procedure)
#Getting the max accuracy among the classifiers both SVC and all MLP's.
max_accuracy = max(accuracy)
 
#Getting the index of max accuracy score.
max_index = [i for i,j in enumerate(accuracy) if j == max_accuracy]
if max_index[0] < hps:
    print('SVM Classifier  with kernel=', hyper_parameters_svm[max_index[0]][0], 'C=',
              hyper_parameters_svm[max_index[0]][1], 'gamma=', hyper_parameters_svm[max_index[0]][2], 'gives the best accuracy score: ', '%.4f'%accuracy[max_index[0]])
else:
    print('MLP Classifier with hidden layer size=', hyper_parameters[max_index[0]-hps][0], 'learning rate=',
          hyper_parameters[max_index[0]-hps][1], 'max iter=', hyper_parameters[max_index[0]-hps][2], 'gives best accuracy score:',
          '%.4f'%accuracy[max_index[0]])
print("---------------------------------")
#Use the classifier that has the maximum accuracy score.(Cross Validation Procedure)
predict = classifiers[max_index[0]].predict(x_test) 
index = 0
accuracy_test = 0
for r in y_test['Target']:
    if predict[index] == r:
        accuracy_test += 1
    index += 1
accuracy_test = accuracy_test / x_test.shape[0]
print('The accuracy score of the test set is :', '%.4f'%accuracy_test)



