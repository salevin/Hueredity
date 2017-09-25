import pickle

import numpy as np
import tensorflow as tf


def main():
    colors = pickle.load(open('colors.p', 'rb'))
    ncolors = len(colors)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=ncolors,
                                            model_dir="./hueredity_model")


    new_samples = np.array(
        [[0,139,137],
         [255, 0, 0],
         [0, 0, 0]], dtype=np.int)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [colors[int(p["classes"][0])] for p in predictions]

    for i, j in zip(new_samples, predicted_classes):
        print("{} is color {}".format(i, j))




if __name__ == "__main__":
    main()
