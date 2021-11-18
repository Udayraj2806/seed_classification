import numpy as np
import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('seeds_dataset.csv')





X = df[['area','perimeter','compactness','lengthOfKernel','widthOfKernel','asymmetryCoefficient','lengthOfKernelGroove','seedType']].values
# y = df['seedType']

def unique_vals(rows,col):
    return set([row[col] for row in rows])




def class_counts(rows):
    counts={}
    for row in rows:
        label=row[-1]
        if label not in counts:
            counts[label]=0
        counts[label]+=1
    return counts



def is_numeric(value):
    return isinstance(value,int) or isinstance(value,float)


class Question:
    def __init__(self,column,value):
        self.column=column
        self.value=value

    def match(self,example):
        val=example[self.column]
        if is_numeric(val):
            return val>=self.value
        else:
            return val==self.value

    def __repr__(self):
        condition = "= ="
        if is_numeric(self.value):
            condition= ">="
        return "Is %s %s %s?" %( X[self.column],condition,str(self.value))

def partition(rows,question):
    true_rows,false_rows =[],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows,false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity =1
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity-=prob_of_lbl**2
    return impurity

def info_gain(left,right,current_uncertainity):
    p=float(len(left))/(len(left)+ len(right))
    return current_uncertainity- p*gini(left)- (1-p)*gini(right)

def find_best_split(rows):
    best_gain = 0
    best_question=None
    current_uncertainty=gini(rows)
    n_features=len(rows[0]) -1

    for col in range(n_features):
        values=set([row[col] for row in rows])
        for val in values:
            question =Question(col,val)
            true_rows,false_rows=partition(rows,question)
            if len(true_rows)==0 or len(false_rows)==0:
                continue
            gain=info_gain(true_rows,false_rows,current_uncertainty)

            if gain>=best_gain:
                best_gain,best_question=gain,question

    return best_gain,best_question


class Leaf:

    def __init__(self,rows):
        self.predictions=class_counts(rows)

class Decision_Node:

    def __init__(self,question,true_branch,fasle_branch):
        self.question=question
        self.true_branch=true_branch
        self.false_branch=fasle_branch

def classify1(row,node):

    if isinstance(node,Leaf):
        return node.predictions


    if node.question.match(row):
        return classify1(row,node.true_branch)
    else:
        return classify1(row,node.false_branch)


class classify:

    def __init__(self,node):
        self.row = []
        self.node=node
        # self.r=classify1(self.p,node)
    def new(self):
        self.r = classify1(self.row,self.node)

    def print_leaf(self):
        total = sum(self.r.values()) * 1.0
        probs = {}
        for lbl in self.r.keys():
            probs[lbl] = str(int(self.r[lbl] / total * 100)) + "%"
        return probs

def build_tree(rows):
    gain,question = find_best_split(rows)

    if gain==0:
        return Leaf(rows)

    true_rows,flase_rows=partition(rows,question)
    true_branch= build_tree(true_rows)
    false_branch=build_tree(flase_rows)

    return Decision_Node(question,true_branch,false_branch)

def print_tree(node,spacing=""):

    if isinstance(node,Leaf):
        print(spacing+ "predict",node.predictions)
        return
    print(spacing+str(node.question))

    print(spacing + '-->True:')
    print_tree(node.true_branch,spacing+ " ")

    print(spacing + '-->False:')
    print_tree(node.false_branch, spacing + " ")

# def classify1(row,node):
#
#     if isinstance(node,Leaf):
#         return node.predictions
#
#
#     if node.question.match(row):
#         return classify1(row,node.true_branch)
#     else:
#         return classify1(row,node.false_branch)

# def print_leaf(counts):
#
#     total =sum(counts.values()) * 1.0
#     probs ={}
#     for lbl in counts.keys():
#         probs[lbl]= str(int(counts[lbl]/total * 100))+ "%"
#     return probs




if __name__ == '__main__':

    my_tree=build_tree(X)

   # print_tree(my_tree)

#     testing_data =[[1.87,588.65,100.8648,100.139,53.463,38.696,5954.967],[
#  14.49,14.61,0.8538,5.715,3.113,4.116,5.396],[
# 14.33,14.28,0.8831,5.504,3.199,3.328,5.224]]
#     for row in testing_data:
#         c=(d.print_leaf(classify1(row,my_tree)))


    d = classify(my_tree)
    r=[10.87, 9.65, 8.8648, 7.139, 6.463, 5.696, 4.967]
    d.row=r
    d.new()
    e=d.print_leaf()
    print(e)




    import pickle
    pickle.dump(d,open('model.pkl','wb'))



# # http://127.0.0.1:5000/
# print(y)
# # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=3)
# # print ('Train set:', X_train.shape,  y_train.shape)
# # print ('Test set:', X_test.shape,  y_test.shape)
#
#
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
#
# knn =  DecisionTreeClassifier()
# knn.fit(X_train,y_train)
#
# y_pred = knn.predict(X_test)
# # print(accuracy_score(y_test,y_pred))
#


# import pickle
# pickle.dump(knn,open('model.pkl','wb'))