import pandas as pd
import numpy as np
import sys
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords


class Classifier:

    def __init__(self):

        self.vocabulary = set()
        self.log_prior = {}
        self.log_likelihood = defaultdict(lambda: defaultdict(float))
        self.labels = None

        self.confusion_matrix = None
        self.accuracy = 0.0
        self.precisions = None

    def train(self, df_train):

        word_label_count = defaultdict(Counter)
        label_freq_dist = Counter()

        #df_train = self.remove_head_tail(df_train)

        if len(sys.argv) == 4 and sys.argv[3] == "--stop":
            df_train = self.remove_stopwords(df_train)

        for index, row in df_train.iterrows():
            label_freq_dist[row["relation"]] += 1
            document = row["tokens"].split()
            #document = row["tokens"]

            for word in document:
                word_label_count[row["relation"]][word] += 1
                self.vocabulary.add(word)

        data_size = len(df_train.index)
        vocab_size = len(self.vocabulary)

        self.labels = label_freq_dist.keys()

        for label in self.labels:
            # Calculate Log Prior
            self.log_prior[label] = np.log2(label_freq_dist[label] / data_size)

            # Calculate Log Likelihood
            for word in self.vocabulary:
                count_wc = word_label_count[label][word]
                count_wpc = sum(word_label_count[label].values())

                self.log_likelihood[label][word] = np.log2((count_wc + 1) / (count_wpc + vocab_size))

    def test(self, df_test):
        df_results = pd.DataFrame(columns=["original_label", "classifier_assigned_label", "row_id"])

        count = 0
        correct = 0

        if (len(sys.argv) > 2 and sys.argv[2] == "--stop") or (len(sys.argv) > 3 and sys.argv[3] == "--stop"):
            df_test = self.remove_stopwords(df_test)

        for index, row in df_test.iterrows():
            document = row["tokens"].split()
            #document = row["tokens"]
            sum_label = {}
            for label in self.labels:
                sum_label[label] = self.log_prior[label]

                for word in document:
                    if word in self.vocabulary:
                        sum_label[label] += self.log_likelihood[label][word]

            pred_label = max(sum_label, key=sum_label.get)

            df_results.loc[index] = [row["relation"], pred_label, row["row_id"]]

            count += 1
            if row["relation"] == pred_label:
                correct += 1

        self.accuracy = correct / count

        return df_results

    def get_precisions(self):
        precision_pub = self.confusion_matrix[0][0] / np.sum(self.confusion_matrix[0])
        precision_dir = self.confusion_matrix[1][1] / np.sum(self.confusion_matrix[1])
        precision_per = self.confusion_matrix[2][2] / np.sum(self.confusion_matrix[2])
        precision_cha = self.confusion_matrix[3][3] / np.sum(self.confusion_matrix[3])

        self.precisions = [precision_pub, precision_dir, precision_per, precision_cha]

        return {"Publisher":precision_pub, "Director": precision_dir,
                "Performer": precision_per, "Characters": precision_cha}

    def get_recalls(self):

        rotated_matrix = np.rot90(self.confusion_matrix, 3)

        recall_pub = rotated_matrix[0][3] / np.sum(rotated_matrix[0])
        recall_dir = rotated_matrix[1][2] / np.sum(rotated_matrix[1])
        recall_per = rotated_matrix[2][1] / np.sum(rotated_matrix[2])
        recall_cha = rotated_matrix[3][0] / np.sum(rotated_matrix[3])

        return {"Publisher": recall_pub, "Director": recall_dir,
                "Performer": recall_per, "Characters": recall_cha}

    def macroaverage_precision(self):

        return np.sum(self.precisions)/4

    def microaverage_precision(self):

        numerator = self.confusion_matrix[0][0] + self.confusion_matrix[1][1]\
                    + self.confusion_matrix[2][2] + self.confusion_matrix[3][3]

        denominator = np.concatenate(self.confusion_matrix).sum()

        return numerator/denominator


    def remove_stopwords(self, df):

        stop_words = stopwords.words('english')
        stop_words.append("\"")
        stop_words.append(".")
        stop_words.append(",")
        stop_words.append("-")

        # This functionality is used for lowercase words
        #documents = df["tokens"].lower().to_list()
        documents = df["tokens"].to_list()
        new_documents = []
        for document in documents:
            document = document.split()
            new_document = []
            for word in document:
                if word not in stop_words:
                    new_document.append(word)

            new_document = " ".join(new_document)
            new_documents.append(new_document)

        df["tokens"] = new_documents

        return df

    def remove_head_tail(self, df):

        new_documents = []

        for index, row in df.iterrows():
            heads = row["head_pos"]
            tails = row["tail_pos"]
            document = row["tokens"]

            heads_list = self.convert_to_int(heads)
            tails_list = self.convert_to_int(tails)
            document = document.split()

            remove_list = heads_list
            remove_list.extend(tails_list)

            for i in sorted(remove_list, reverse=True):
                try:
                    del document[i]
                except:
                    break

            document = " ".join(document)
            new_documents.append(document)

        df["tokens"] = new_documents

        return df

    def convert_to_int(self, temp_str):

        temp_list = temp_str.split()
        new_list = []
        for item in temp_list:
            new_list.append(int(item))

        return new_list










