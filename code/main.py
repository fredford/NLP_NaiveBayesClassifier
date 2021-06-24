import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import confusion_matrix

from classifier import Classifier
from k_fold import KFold


def main():
    # The four classes established for the dataset
    classes = ["Publisher", "Director", "Performer", "Characters"]

    # Get the directory for the current file and project directory
    file_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

    # Create pandas DataFrames for the training and test files
    df_train = pd.read_csv(os.path.join(project_path, "data/train.csv"))
    df_test = pd.read_csv(os.path.join(project_path, "data/test.csv"))
    df_dev = pd.read_csv(os.path.join(project_path, "data/eval.csv"))

    # Train the Naive Bayes model
    if (len(sys.argv) > 2 and sys.argv[2] == "--epochs") or (len(sys.argv) > 3 and sys.argv[3] == "--epochs"):
        epochs = 3
    else:
        epochs = 1

    #best_classifier = Classifier()
    #best_classifier.train(df_train)
    #best_classifier.test(df_test)
    #best_accuracy = best_classifier.accuracy
    best_accuracy = 0.0

    count = 0
    sum_accuracy = 0.0

    for e in range(epochs):
        kfold = KFold()
        kfold.k_fold_prep(df_train)
        for i in range(3):
            kfold.train(i)
            count += 1
            sum_accuracy += kfold.best_accuracy

        if kfold.best_accuracy > best_accuracy:
            best_accuracy = kfold.best_accuracy
            best_classifier = kfold.best_classifier

    print("Training Accuracy: " + str(sum_accuracy/count))

    # Test the Naive Bayes model on the test set
    df_output_test = best_classifier.test(df_test)

    y_true = df_output_test["original_label"]
    y_pred = df_output_test["classifier_assigned_label"]
    labels = list(best_classifier.labels)

    best_classifier.confusion_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # If a test parameter is not specified as a commandline argument display the output parameters
    if len(sys.argv) > 1 and sys.argv[1] == "--verbose":
        # Get the precision and recall values calculated by the classifier
        precision_dict = best_classifier.get_precisions()
        recall_dict = best_classifier.get_recalls()

        # Create the confusion matrix dataframe
        df_confusion_matrix = pd.DataFrame(best_classifier.confusion_matrix,
                                           index=["Publisher", "Director", "Performer", "Characters"],
                                           columns=["Publisher", "Director", "Performer", "Characters"])

        print("Testing accuracy = {:.4f}".format(best_classifier.accuracy))

        for type in classes:
            print("{0}\tPrecision: {1:1.6f} {0}\tRecall: {2:2.6f}"
                  .format(type, precision_dict[type], recall_dict[type]))

        print("Macroaverage Precision: {0:2.6f}".format(best_classifier.macroaverage_precision()))
        print("Microaverage Precision: {0:2.6f}".format(best_classifier.microaverage_precision()))

        # Generate a confusion matrix plot
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_confusion_matrix, annot=True, cmap='Blues', )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    # Test the Naive Bayes model on the dev set
    df_output_dev = best_classifier.test(df_dev)

    # Write the pandas DataFrames to csv files without pandas indexing
    df_output_dev.to_csv(os.path.join(project_path, "output/output_dev.csv"), index=False)
    df_output_test.to_csv(os.path.join(project_path, "output/output_test.csv"), index=False)


main()
