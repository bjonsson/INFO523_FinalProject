# *Swapping out the "malware_detector.py" ExtraTreesClassifier for SVM

# Following along to this tutorial by Cloud and ML Online:
# https://www.youtube.com/watch?v=7sz4WpkUIIs

# This program detects malware based on the file's PE headers. PE files can only be safely examined on an
# isolated machine, like a virtual machine.

import pandas as pd

# The first 41,323 files/rows are legitimate, whereas the next 96,724 files/rows are from virusshare.com.
malData = pd.read_csv("MalwareData.csv", sep= "|")

# We're dropping the "legitimate" column and separating the malware from the legitimate files
legit = malData[0:50].drop(['Name', 'md5', "legitimate"], axis=1)  # Axis 1 refers to columns. 0 refers to rows.
mal = malData[50::].drop(['Name', 'md5', "legitimate"], axis=1)

too_slow1 = malData.head(50)
too_slow2 = malData.tail(50)
too_slow = 	pd.concat([too_slow1, too_slow2])

print(too_slow)

# SVM Swapout

#Dropping the columns worked. There were originally 57 columns.
print(legit.head())

#Remove columns from overall data set as well.
#Old way
data_in = too_slow.drop(['Name', 'md5', 'legitimate'], axis=1).values
labels = too_slow['legitimate'].values

from sklearn.model_selection import train_test_split

legit_train, legit_test, mal_train, mal_test = train_test_split(data_in, labels, test_size=0.2)



from sklearn import svm

classifier = svm.SVC(kernel='linear', gamma='auto')
# reads in the data and prepares the model
classifier.fit(legit_train, mal_train)

# y = f(x)
mal_predict = classifier.predict(legit_test)

# # To read report:
# #   precision = true positive/ (true positive + false positive)
# #   support = the quantity of each type of record
# from sklearn.metrics import classification_report
# print(classification_report(mal_test, mal_predict))

print("The percentage of accuracy of the model: ", classifier.score(legit_test, mal_test)*100)

# Model is finished.




# Testing for false positives/negatives
from sklearn.metrics import confusion_matrix

# Running a confusion matrix. A confusion matrix compares actual values with predicted values.
# .predict() predicts the class for new data instances using the classification model

conf_mat = confusion_matrix(mal_test,mal_predict)

# We are expecting a 2x2 matrix (standard size for Python confusion matrix function)
print(conf_mat.shape)
# Confirming that this is a multidimensional array
print(type(conf_mat))

# We can see that the malware errors were on top, and the legit data's errors were on the bottom
print(conf_mat)
# We selected values that were "True" for each data set, so ones that were not malware are on top,
# and ones there were malware and got through are on the bottom.
print("Percentage of false positives: ",conf_mat[0][1]/sum(conf_mat[0]*100))
print("Percentage of false negatives: ",conf_mat[1][0]/sum(conf_mat[1]*100))