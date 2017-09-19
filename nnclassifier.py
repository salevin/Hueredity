import csv

import numpy as np
import tensorflow as tf


def formatData(csvfile):
    colors = {}
    num = 0
    with open('traindata.csv', 'w', newline='') as trainfile:
        wr = csv.writer(trainfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        with open(csvfile, newline='') as prevfile:
            next(prevfile)
            prevread = csv.reader(prevfile, delimiter=',')
            for row in prevread:
                if row[3] not in colors:
                    colors[row[3]] = num
                    num += 1
                row[3] = colors[row[3]]
                nrow = [int(i) for i in row]
                wr.writerow(nrow)
    return {v: k for k, v in colors.items()}


def main():
    DEFAULT_FILE = "satfaces.csv"
    NEW_FILE = "traindata.csv"

    colors = formatData(DEFAULT_FILE)
    ncolors = len(colors)

    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=NEW_FILE,
        target_dtype=np.int,
        features_dtype=np.int)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=ncolors,
                                            model_dir="/tmp/hueredity_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True)

    classifier.train(input_fn=train_input_fn, steps=2000)

    accuracy_score = classifier.evaluate(input_fn=train_input_fn)["accuracy"]

    print("\nTrain Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == '__main__':
    main()
