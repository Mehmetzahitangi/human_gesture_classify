#import libraries for KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#The dataset provides the patients’ information
df_0 = pd.read_csv("0.csv",header=None)
df_0.info(verbose = True)
df_1 = pd.read_csv("1.csv",header=None)
df_1.info(verbose = True)
df_2 = pd.read_csv("2.csv",header=None)
df_2.info(verbose = True)
df_3 = pd.read_csv("3.csv",header=None)
df_3.info(verbose = True)

concat_data = [df_0, df_1, df_2, df_3 ]
#result = df_0.append(df_1)
df = pd.concat(concat_data)

#%% 

y = df[64].values
X = df.drop(64,axis=1).values

#Train Test Split data==> 70% of data set for Train, 30% of data set for Test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


knn = KNeighborsClassifier(n_neighbors=5)


# 5-fold cross-validation on the training set.
sfs1 = SFS(knn, 
           k_features=12, 
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)
sfs1 = sfs1.fit(X_train, y_train)

print('Selected features:', sfs1.k_feature_idx_)


X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
knn.fit(X_train_sfs, y_train)
y_pred = knn.predict(X_test_sfs)

# Compute the accuracy of the prediction
acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
print('Test set accuracy: %.2f %%' % (acc * 100))


f1_score_weighted = f1_score(y_test, y_pred, average='weighted')
print("f1_score_weighted: ", f1_score_weighted)

f1_score_macro = f1_score(y_test, y_pred, average='macro')
print("f1_score_macro: ", f1_score_macro)
f1_score_micro = f1_score(y_test, y_pred, average='micro')
print("f1_score_micro: ", f1_score_micro)
f1_score_ = f1_score(y_test, y_pred, average=None)
print("f1_score: ", f1_score_)


print(classification_report(y_test,y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))

#%% Decision Trees

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth =15, random_state = 42)

# Select the "best" three features via
# 5-fold cross-validation on the training set.

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs2 = SFS(clf, 
           k_features=20, 
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)
sfs2 = sfs2.fit(X_train, y_train)

print('Selected features:', sfs2.k_feature_idx_)

# Generate the new subsets based on the selected features
# Note that the transform call is equivalent to
# X_train[:, sfs1.k_feature_idx_]

X_train_sfs2 = sfs2.transform(X_train)
X_test_sfs2 = sfs2.transform(X_test)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
clf.fit(X_train_sfs2, y_train)
y_pred2 = clf.predict(X_test_sfs2)

# Compute the accuracy of the prediction
acc = float((y_test == y_pred2).sum()) / y_pred2.shape[0]
print('Test set accuracy: %.2f %%' % (acc * 100))


f1_score2 = f1_score(y_test, y_pred2, average='weighted')
print("f1_score_weighted: ", f1_score2)

f1_score_macro2 = f1_score(y_test, y_pred2, average='macro')
print("f1_score_macro: ", f1_score_macro2)
f1_score_micro2 = f1_score(y_test, y_pred2, average='micro')
print("f1_score_micro: ", f1_score_micro2)
f1_score_2 = f1_score(y_test, y_pred2, average=None)
print("f1_score: ", f1_score_2)


print(classification_report(y_test,y_pred2))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred2))

#%%
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# Select the "best" three features via
# 5-fold cross-validation on the training set.

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs3 = SFS(gnb, 
           k_features=10, 
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)
sfs3 = sfs3.fit(X_train, y_train)

print('Selected features:', sfs3.k_feature_idx_)



X_train_sfs3 = sfs3.transform(X_train)
X_test_sfs3 = sfs3.transform(X_test)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
gnb.fit(X_train_sfs3, y_train)
y_pred3 = gnb.predict(X_test_sfs3)

# Compute the accuracy of the prediction
acc = float((y_test == y_pred3).sum()) / y_pred3.shape[0]
print('Test set accuracy: %.2f %%' % (acc * 100))


f1_score3 = f1_score(y_test, y_pred3, average='weighted')
print("f1_score_weighted: ", f1_score3)

f1_score_macro3 = f1_score(y_test, y_pred3, average='macro')
print("f1_score_macro: ", f1_score_macro3)
f1_score_micro3 = f1_score(y_test, y_pred3, average='micro')
print("f1_score_micro: ", f1_score_micro3)
f1_score_3 = f1_score(y_test, y_pred3, average=None)
print("f1_score: ", f1_score_3)


print(classification_report(y_test,y_pred3))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred3))