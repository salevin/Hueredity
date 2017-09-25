import numpy as np
import tensorflow as tf


def main():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]
    ncolors = 28

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=ncolors,
                                            model_dir="./hueredity_model")






if __name__ == "__main__":
    main()
