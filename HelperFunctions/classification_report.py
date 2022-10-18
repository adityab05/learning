"""
Print the classification report using y_test and y_pred as input
"""

# imports
from sklearn.metrics import classification_report

# print classification report
def print_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    
# usage
print_classification_report(y_test, y_pred)