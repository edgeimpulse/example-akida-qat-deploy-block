
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss, mean_squared_error

def predict(model, validation_dataset):
    """Runs a TensorFlow Lite model across a set of inputs"""

    return model.predict(validation_dataset)

def evaluate(model, dataset, Y_test, num_classes):
    prediction = predict(model, dataset)

    Y_labels = []
    for ix in range(num_classes):
        Y_labels.append(ix)
    matrix = confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1), labels=Y_labels)
    report = classification_report(Y_test.argmax(axis=1), prediction.argmax(axis=1), output_dict=True, zero_division=0)

    accuracy = report['accuracy']
    f1 = report['weighted avg']['f1-score']
    loss = log_loss(Y_test, prediction)
    return report, accuracy, f1, loss
