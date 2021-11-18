
import pandas as pd

df = pd.read_csv('seeds_dataset.csv')

X = df[['area','perimeter','compactness','lengthOfKernel','widthOfKernel','asymmetryCoefficient','lengthOfKernelGroove']].values
y = df['seedType']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3,random_state=3)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)


from sklearn import tree
clf = tree.DecisionTreeClassifier()
from sklearn.metrics import accuracy_score


# Train and Classify
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

import pickle
pickle.dump(clf,open('model.pkl','wb'))