# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:24:27 2019

@author: Harshita Rastogi
"""


# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz

def partition(x):
    dic={}
    v=np.unique(x)
    for i in range(0,len(v)):
        indices = [j for j,value in enumerate(x) if value == v[i]]
        dic[v[i]] = indices
    return dic
    raise Exception('Function not yet implemented!')
    
def entropy(y):
    n = 0
    final_entropy = 0
    for i in y.keys():
        n +=len(y[i])
    for j in y.keys():
            entropyy = (len(y[j])/n) * (np.log2(len(y[j])/n))
            final_entropy = final_entropy - entropyy
    return final_entropy             
    
    raise Exception('Function not yet implemented!')
    
def mutual_information(x, y):
    y_entropy = entropy(partition(y))
    x_val = partition(x)
    for i in x_val.values():
        d = []
        for j in i:
            d.append(y[j])
        y_entropy-=(len(i)/len(y) * entropy(partition(d)))
    return(y_entropy)

    raise Exception('Function not yet implemented!')
    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    
    root = {}
    
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for index in range(0, x.shape[1]):
            for value in np.unique(x[:, index]):
                attribute_value_pairs.append((index, value))

    val_y, cnt_y = np.unique(y, return_counts=True)
    if len(np.unique(y)) & (y[0]==1 or y[0]==0):
        return val_y[0]
    
    if(len(attribute_value_pairs) == 0 or depth == max_depth):
        return val_y[np.argmax(cnt_y)]

    MI_attribute_value_pair = []
    for (index, value) in attribute_value_pairs:
        MI_attribute_value_pair.append(mutual_information( x[:, index]==value, y))
    (bestattr, bestvalue) = attribute_value_pairs[np.argmax(MI_attribute_value_pair)]

    attribute_value_pairs_new = attribute_value_pairs.copy()
    attribute_value_pairs_new.remove((bestattr, bestvalue))
    
    currentNodePartition  = partition(x[:, bestattr] == bestvalue)

    for label in currentNodePartition.keys():
        x_subset = x.take(currentNodePartition[label], axis=0)
        y_subset = y.take(currentNodePartition[label], axis=0)
        nodeDecision = bool(label)

        root[(bestattr, bestvalue, nodeDecision)] = id3(x_subset, y_subset, attribute_value_pairs=attribute_value_pairs_new, max_depth=max_depth, depth=depth+1)

    return root
    raise Exception('Function not yet implemented!')

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if not tree:
        return 0
    
    if not isinstance(tree, dict):
        return tree
    for i in tree.keys():
        if (x[i[0]] == i[1]) == i[2]:
            return predict_example(x, tree[i])

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    returns the error = (1/n) * sum(y_true != y_pred)
    """
    n = len(y_true)
    error = 1/n * (np.sum(y_true != y_pred))
    return error


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    output_dir = './Console_Output/PartA/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    M = np.genfromtxt('./monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)

    render_dot_file(dot_str, output_dir + 'partA_my_learned_tree')

    # Compute the test error
    print(decision_tree)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print(y_pred)
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print('Test Accuracy = {0:4.2f}%.'.format((1- tst_err) * 100))

       



     
    
                