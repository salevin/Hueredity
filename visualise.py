import sys

from PyQt5.QtWidgets import QApplication, QWidget, QColorDialog, QMessageBox
from PyQt5.QtGui import QFont
import pickle

import numpy as np
import tensorflow as tf


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Hueredity Color Picker'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200

        self.initClassifier()

        while openColorDialog(self):
            pass

        sys.exit()

    def initClassifier(self):
        self.colors = pickle.load(open('colors.p', 'rb'))
        ncolors = len(self.colors)

        feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]

        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 20, 10],
            n_classes=ncolors,
            model_dir="./hueredity_model")

    def returnColor(self, color):
        new_samples = np.array(
            [color], dtype=np.int)

        pred_in = tf.estimator.inputs.numpy_input_fn(
            x={"x": new_samples},
            num_epochs=1,
            shuffle=False)

        pred = list(self.classifier.predict(input_fn=pred_in))[0]
        pred_color = self.colors[int(pred["classes"][0])]

        msg = QMessageBox()
        msg.setText("{} is the color {}".format(color, pred_color))
        font = QFont()
        font.setPixelSize(20)
        msg.setFont(font)
        msg.exec_()


def openColorDialog(self):
    color = QColorDialog.getColor()

    if color.isValid():
        self.returnColor(color.getRgb()[:3])
        return True
    else:
        return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
