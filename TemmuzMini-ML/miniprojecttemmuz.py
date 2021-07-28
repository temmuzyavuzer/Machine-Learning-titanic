import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Reading  titanic data set from kaggle
# https://www.kaggle.com/c/titanic/overview
titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
test_given_predict = pd.read_csv("gender_submission.csv")

# Get the survived people,Take off the survided people,Add to set together
# Take off the useless parts,Fill empty values for valid results
survived_people = titanic_train["Survived"].values
titanic_train = titanic_train.drop(["Survived"], axis=1)
all_datas = pd.concat([titanic_train, titanic_test], ignore_index=True, axis=0)
all_datas = all_datas.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)
all_datas["Embarked"].fillna(method='ffill', inplace=True)
all_datas["Age"].fillna(all_datas["Age"].median(), inplace=True)

# Encode to temp var
# Avoid temp var multi connection
gender_temp = pd.get_dummies(all_datas['Sex'], prefix='Sex')
gender_temp = gender_temp.iloc[:, 0]
aboard_temp = pd.get_dummies(all_datas['Embarked'], prefix='Embarked')
aboard_temp = aboard_temp.iloc[:, :2]

#Take of col from set, add the new ones
all_datas = all_datas.drop(["Sex", "Embarked"], axis=1)
all_datas = pd.concat([gender_temp, aboard_temp, all_datas], axis=1)
indept_var = all_datas.iloc[:].values

# Divide set into train,test
indept_var_train = indept_var[:891, :]
indept_var_test = indept_var[891:, :]
survived_people_train = survived_people[:]
survived_people_test = test_given_predict.iloc[:, 1].values

# Characteristic Scaling
scaling = StandardScaler()
indept_var_train = scaling.fit_transform(indept_var_train)
indept_var_test = scaling.transform(indept_var_test)

# Fit classifier
# Predict,Create confusion matrix
classifier_chebyshev = KNeighborsClassifier(n_neighbors=13, metric='chebyshev', p=7, n_jobs=-1)
classifier_chebyshev.fit(indept_var_train, survived_people_train)
survived_people_predict = classifier_chebyshev.predict(indept_var_test)
matrix = confusion_matrix(survived_people_test, survived_people_predict)

# Required values for acc,recall,prec
False_Positive = matrix[1][0]
False_Negative = matrix[0][1]
True_Positive = matrix[0][0]
True_Negative = matrix[1][1]

accuracy = ((True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative))
precision = True_Positive / (True_Positive + False_Positive)
recall = True_Positive / (True_Positive + False_Negative)
f_score = (2 * precision * recall) / (precision + recall)

print("Chebyshev Results")
print("Accuracy is {:0.6f}.".format(accuracy))
print("Precision is {:0.6f}.".format(precision))
print("Recall is {:0.6f}.".format(recall))
print("Fscore is {:0.6f}.\n".format(f_score))

# New csv file to see the result for each person
isSurvived = {'Survived':survived_people_predict}
idPeople = {'PassengerId': titanic_test.iloc[:, 0].values}
alldata_predicts = pd.concat([pd.DataFrame(idPeople), pd.DataFrame(isSurvived)], axis=1)
alldata_predicts.to_csv('chebyshev_predictions.csv', encoding='utf-8', index=False)

# Principal Component Analysis (PCA)
pca = PCA(n_components=4)
indept_var_train = pca.fit_transform(indept_var_train)
indept_var_test = pca.transform(indept_var_test)
variance = pca.explained_variance_ratio_



#***************SECOND EXPERIMENT**************
# Fit classifier,# Predict,Create confusion matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

classifier_minkowski = KNeighborsClassifier(n_neighbors=13, metric='minkowski', p=3, n_jobs=-1)
classifier_minkowski.fit(indept_var_train, survived_people_train)
survived_people_predict = classifier_minkowski.predict(indept_var_test)
matrix = confusion_matrix(survived_people_test, survived_people_predict)

# Required values for acc,recall,prec
True_Positive = matrix[0][0]
True_Negative = matrix[1][1]
False_Positive = matrix[1][0]
False_Negative = matrix[0][1]

accuracy = ((True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative))
precision = True_Positive / (True_Positive + False_Positive)
recall = True_Positive / (True_Positive + False_Negative)
f_score = (2 * precision * recall) / (precision + recall)

print("Minkowski Results")
print("Accuracy is {:0.6f}".format(accuracy))
print("Precision is {:0.6f}".format(precision))
print("Recall is {:0.6f}".format(recall))
print("Fscore is {:0.6f}\n".format(f_score))

# New csv file to see the result for each person
isSurvived = {'Survived':survived_people_predict}
idPeople = {'PassengerId': titanic_test.iloc[:, 0].values}
alldata_predicts = pd.concat([pd.DataFrame(idPeople), pd.DataFrame(isSurvived)], axis=1)
alldata_predicts.to_csv('minkowski_predictions.csv', encoding='utf-8', index=False)

# Importing the datasets
titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
test_given_predict = pd.read_csv("gender_submission.csv")

# Get the survived people,Take off the survided people,Add to set together
# Take off the useless parts,Fill empty values for valid results
survived_people = titanic_train["Survived"].values
titanic_train = titanic_train.drop(["Survived"], axis=1)
all_datas = pd.concat([titanic_train, titanic_test], ignore_index=True, axis=0)
all_datas = all_datas.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)
all_datas["Embarked"].fillna(method='ffill', inplace=True)
all_datas["Age"].fillna(all_datas["Age"].median(), inplace=True)

# Encode to temp var
# Avoid temp var multi connection
gender_temp = pd.get_dummies(all_datas['Sex'], prefix='Sex')
gender_temp = gender_temp.iloc[:, 0]
aboard_temp = pd.get_dummies(all_datas['Embarked'], prefix='Embarked')
aboard_temp = aboard_temp.iloc[:, :2]

#Take of col from set, add the new ones
all_datas = all_datas.drop(["Sex", "Embarked"], axis=1)
all_datas = pd.concat([gender_temp, aboard_temp, all_datas], axis=1)

indept_var = all_datas.iloc[:].values

# Divide set into train,test
indept_var_train = indept_var[:891, :]
indept_var_test = indept_var[891:, :]
survived_people_train = survived_people[:]
survived_people_test = test_given_predict.iloc[:, 1].values

# Characteristic Scaling
from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
indept_var_train = scaling.fit_transform(indept_var_train)
indept_var_test = scaling.transform(indept_var_test)

#***************THIRD EXPERIMENT**************
# Fit classifier
# Predict,Create confusion matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

classifier_manhattan = KNeighborsClassifier(n_neighbors=13, metric='manhattan', p=1, n_jobs=-1)
classifier_manhattan.fit(indept_var_train, survived_people_train)
survived_people_predict = classifier_manhattan.predict(indept_var_test)
matrix = confusion_matrix(survived_people_test, survived_people_predict)

# Required values for acc,recall,prec
True_Positive = matrix[0][0]
True_Negative = matrix[1][1]
False_Positive = matrix[1][0]
False_Negative = matrix[0][1]

accuracy = ((True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative))
precision = True_Positive / (True_Positive + False_Positive)
recall = True_Positive / (True_Positive + False_Negative)
f_score = (2 * precision * recall) / (precision + recall)

print("Manhattan Results")
print("Accuracy is {:0.6f}".format(accuracy))
print("Precision is {:0.6f}".format(precision))
print("Recall is {:0.6f}".format(recall))
print("Fscore is {:0.6f}\n".format(f_score))

# New csv file to see the result for each person
isSurvived = {'Survived':survived_people_predict}
idPeople = {'PassengerId': titanic_test.iloc[:, 0].values}
alldata_predicts = pd.concat([pd.DataFrame(idPeople), pd.DataFrame(isSurvived)], axis=1)
alldata_predicts.to_csv('manhattan_predictions.csv', encoding='utf-8', index=False)

# Principal Component Analysis (PCA)
pca = PCA(n_components=4)
indept_var_train = pca.fit_transform(indept_var_train)
indept_var_test = pca.transform(indept_var_test)
variance = pca.explained_variance_ratio_

#***************FORTH EXPERIMENT**************
# Fit classifier
# Predict,Create confusion matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

classifier_euclidean = KNeighborsClassifier(n_neighbors=13, metric='euclidean', p=2, n_jobs=-1)
classifier_euclidean.fit(indept_var_train, survived_people_train)
survived_people_predict = classifier_euclidean.predict(indept_var_test)
matrix = confusion_matrix(survived_people_test, survived_people_predict)

# Required values for acc,recall,prec
True_Positive = matrix[0][0]
True_Negative = matrix[1][1]
False_Positive = matrix[1][0]
False_Negative = matrix[0][1]

accuracy = ((True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative))
precision = True_Positive / (True_Positive + False_Positive)
recall = True_Positive / (True_Positive + False_Negative)
f_score = (2 * precision * recall) / (precision + recall)

print("Euclidean Results")
print("Accuracy is {:0.6f}".format(accuracy))
print("Precision is {:0.6f}".format(precision))
print("Recall is {:0.6f}".format(recall))
print("Fscore is {:0.6f}\n".format(f_score))

# New csv file to see the result for each person
isSurvived = {'Survived':survived_people_predict}
idPeople = {'PassengerId': titanic_test.iloc[:, 0].values}
alldata_predicts = pd.concat([pd.DataFrame(idPeople), pd.DataFrame(isSurvived)], axis=1)
alldata_predicts.to_csv('euclidean_predictions.csv', encoding='utf-8', index=False)