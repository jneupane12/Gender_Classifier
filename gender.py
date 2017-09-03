from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # accuracy_score is a method which returns "accuray classification score"
import numpy as np


# [height],[weight],[shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], 
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]


Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#classifiers
clf=tree.DecisionTreeClassifier()
neigh=KNeighborsClassifier(n_neighbors=3)
gab=GaussianNB()
forest=RandomForestClassifier(n_estimators=2)

# here we train the models
clf=clf.fit(X, Y)
neigh=neigh.fit(X,Y)
gab=gab.fit(X,Y)
forest=forest.fit(X,Y)




# prediction using given data
prediction = clf.predict([[173,60,41]])   
predictionNiegh= neigh.predict([[173,60,41]])   
predictionGab=gab.predict([[173,60,41]])  
predictionForest=forest.predict ([[173,60,41]])  






# printing the result 
print ("decision tree prediction is", prediction)
print ('nigh prediction is',predictionNiegh)
print ('gab prediction is',predictionGab)
print('forest prediction is',predictionForest)




# Testing with the same data

pred_tree=clf.predict(X)
acc_tree=accuracy_score(Y,pred_tree)*100
print ("\nAccuracy for decision tree:{}".format(acc_tree))


pred_neigh=neigh.predict(X)
acc_neigh=accuracy_score(Y,pred_neigh)*100
print ("Accuracy for KNeighbourClassifier is :{}".format(acc_neigh))

pred_gauss=gab.predict(X)
acc_gauss=accuracy_score(Y,pred_gauss)*100
print ("Accuracy for GaussianNB is:{}".format(acc_gauss))


pred_forest=forest.predict(X)
acc_forest=accuracy_score(Y,pred_forest)*100
print ("Accuracy for Random Forest Classifier is :{}".format(acc_forest))


#finding the best classifier
index = np.argmax([acc_tree,acc_neigh,acc_gauss,acc_forest])
classifiers = {0:"Tree",1:"Neigh",3:"Gauss",4:"Forest"}
print ("\nBest gender classifier is: {}".format(classifiers[index]))





