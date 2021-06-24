import pandas as pd
import numpy as np
import sys
from collections import defaultdict, Counter
from classifier import Classifier


class KFold:

    def __init__(self):

        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                           "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'himself', "she's", 'herself',
                           'it', 'itself', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                           'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at', 'for', 'with', 'about', 'against', 'between',
                           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                           'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                           'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                           'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                           'very', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                           'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                           "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                           "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                           "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                           'wouldn', "wouldn't"]

        self.best_accuracy = 0.0
        self.best_classifier = None
        self.df_1 = None
        self.df_2 = None
        self.df_3 = None

    def k_fold_prep(self, df):

        random_df = df.sample(frac = 1)

        if len(random_df) % 3 == 0:
            fold_size_1 = len(random_df) / 3
            fold_size_2 = len(random_df) / 3

        elif len(random_df) % 3 == 1:
            fold_size_1 = len(random_df) // 3
            fold_size_2 = len(random_df) // 3

        else:
            fold_size_1 = len(random_df) // 3
            fold_size_2 = len(random_df) // 3 + 1

        index_1 = fold_size_1
        index_2 = index_1 + fold_size_2

        self.df_1 = random_df.iloc[:index_1]
        self.df_2 = random_df.iloc[index_1:index_2]
        self.df_3 = random_df.iloc[index_2:]

    def train(self, n):

        if n == 0:
            test_df = self.df_1
            train_df = self.df_2.append(self.df_3).reset_index()
        elif n == 1:
            test_df = self.df_2
            train_df = self.df_1.append(self.df_3).reset_index()
        elif n == 2:
            test_df = self.df_3
            train_df = self.df_2.append(self.df_1).reset_index()

        classifier = Classifier()
        classifier.train(train_df)
        classifier.test(test_df)
        accuracy = classifier.accuracy
        if len(sys.argv) >= 2 and sys.argv[1] == "--verbose":
            print("Training iteration {} accuracy = {:.4f}%".format(n+1, accuracy*100))

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_classifier = classifier

