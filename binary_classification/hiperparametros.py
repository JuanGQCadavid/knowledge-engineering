import sys
import pandas as pandas
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as numpy
import seaborn as sns
import graphviz
import itertools
from sklearn import linear_model
from sklearn import tree
from tabulate import tabulate
from sklearn.model_selection import train_test_split


best_model = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar',  'pH',  'sulphates',  'alcohol']


def hiperparametros_test(features, label):

    x_train, x_validation, y_train, y_validation = train_test_split(
        features, label, test_size=0.10)
    results_validation = []
    results_train = []

    for i in range(1, 21):
        
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(x_train.values, y_train.values.ravel())

        score_validation = clf.score(
            x_validation.values, y_validation.values.ravel())
        score_train = clf.score(x_train.values, y_train.values.ravel())
        results_validation.append(score_validation)
        results_train.append(score_train)
    return results_validation, results_train


def main():
    df = pandas.read_csv("whitewine.csv", sep=';')
    label = df[['quality']].copy()
    label = label.applymap(lambda x: 1 if x > 5 else 0)
    features = df[best_model].copy()
    results_validation, results_train = hiperparametros_test(features, label)

    pylab.plot(range(1, 21), results_validation, '-rs', label='validation score')
    pylab.plot(range(1, 21), results_train, '-bs', label='train score')
    pylab.legend(loc='upper left')
    pylab.show()

def tree_classifier(features_train, labels_train, deep):
    clf = tree.DecisionTreeClassifier(max_depth=deep)
    clf = clf.fit(features_train, labels_train)

    '''dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render("iris")
    graph'''
    return clf


main()
