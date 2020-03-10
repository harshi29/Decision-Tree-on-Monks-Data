# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:38:25 2019

@author: Harshita Rastogi
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix  
import os
import numpy as np
from sklearn import tree

def prediction(Xtst, entropy):
    y_pred = entropy.predict(Xtst)
    return y_pred

def evaluate_confusion_matrix(y_test, y_pred, cm_dir):
    confusion_matrix_results = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix :')
    print(confusion_matrix_results)
    
def train_classifier(Xtrn, Xtst, ytrn,y_test, i, cm_file_name):
    classifier = DecisionTreeClassifier(max_depth = i)
    classifier = classifier.fit(Xtrn, ytrn)
    y_pred = prediction(Xtst, classifier)
    evaluate_confusion_matrix(y_test, y_pred, cm_file_name)
    return classifier

if __name__ == '__main__':
    # Load the training data
    output_dir = './Output/PartC/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    M = np.genfromtxt('./monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    depth = [1,3,5]

    tstAcc= {}
    for i in depth:
        cm_dir = output_dir + "PartC" + str(i)
        classifier = train_classifier(Xtrn, Xtst, ytrn, ytst, i, cm_dir)
        dotfile = open(output_dir+ "PartC.dot"+ str(i), 'w')
        dot_file_name = output_dir +"PartC.dot"+ str(i)
        final_class = []
        class_list = np.unique(ytrn)
        for class_name_item in class_list:
            final_class.append(str(class_name_item))
        tree.export_graphviz(classifier, out_file=dotfile ,class_names= final_class)
        dotfile.close()
    
        accuracy = classifier.score(Xtst, ytst)
        print("Error is :", 100 - accuracy * 100, "%")
        print("Accuracy is:", accuracy * 100, "%")
        output_file_name = output_dir + "PartC_tree" + str(i)
        tstAcc[i] = accuracy
        cwd = os.getcwd()
        from subprocess import check_call
        check_call(['dot', '-Tpng', dot_file_name , '-o', output_file_name + '.png'])
   
    print("Accuracy for each depth-:\n")
    for depth, accuracy_value in tstAcc.items():
        tst = tstAcc[depth] * 100
        print("Test Accuracy at depth-" + str(depth) + ": ", tst)