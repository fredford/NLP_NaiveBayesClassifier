# Naive Bayes Classifier

_Work in progress, requires changes to speed up computation time involving simplifying the process of using Pandas_

This is an implementation of a Naive Bayes classifier that uses bag-of-words features to classify the relation label of a given sentence. The program classifies a sentence based upon the following relations given some piece of text:

1. Publisher
2. Director
3. Performer
4. Characters

## Data

The assignment's training data can be found in [data/train.csv](data/train.csv).

## 1. Members

- Fraser Redford
- Scott Kavalinas

## 2. Resources

- https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
- https://web.stanford.edu/~jurafsky/slp3/4.pdf
- https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
- https://machinelearningmastery.com/k-fold-cross-validation/
- https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
- https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

## 3. Installation

- Python3
- Pandas
- NLTK
- Seaborn
- Numpy
- Matplotlib
- Scikit Learn

The following packages are required to be installed:

```
$ pip3 install pandas
$ pip3 install nltk
$ pip3 install seaborn
$ pip3 install numpy
$ pip3 install matplotlib
$ pip3 install scikit-learn
```

We used SciKit Learn for generating a confusion matrix and nothing else in the program.

## 4. Running

The program can be run using the following functionalities:

- `--verbose` provides a display of the results
- `--stop` runs the classification process using stopwords
- `--epochs` performs 3 instances of the 3-Fold Cross-Validation taking the best results

In the main directory for the program the following commands can be run to execute the program

```
$ python3 code/main.py
```

This will generate the output files with no display of results.

```
$ python3 code/main.py --verbose
```

This will generate the output files with a display of results.

```
$ python3 code/main.py --verbose --stop
```

This will run the program and generate the output files with a display of results and using the stop words

```
$ python3 code/main.py --verbose --epochs
```

This will run the program and generate the output files with a display of results and using 3 epochs

```
$ python3 code/main.py --verbose --epochs --stop
```

This will generate the output files, display the results, use 3 epochs and the stop words.
