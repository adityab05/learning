"""
Prints a seaborn heatmap as a confusion matrix in Jupyter Notebook
"""

# imports
from sklearn.metrics import confusion_matrix

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn


# function that plots the confusion matrix
def plot_confusion_matrix(y_test, y_pred):

    # get confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # plot the confusion matrix
    plt.figure(figsize=(20, 15))
    sn.heatmap(confusion_mat, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()


# Usage
plot_confusion_matrix(y_test, y_pred)