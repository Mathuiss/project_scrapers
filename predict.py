import numpy as np
import sys
from keras.models import load_model
from keras.models import Sequential


def predict(name):
    model = load_model("models/small_model4.h5")

    x_train = np.load(f"models/{name}x_train.npy")
    x_test = np.load(f"models/{name}x_test.npy")
    x = np.append(x_train, x_test).reshape(-1, 32)

    y_train = np.load(f"models/{name}y_train.npy")
    y_test = np.load(f"models/{name}y_test.npy")
    y = np.append(y_train, y_test).reshape(-1, 1)

    correct = 0
    fault = 0

    arr_pred = model.predict(x)

    for i in range(len(arr_pred)):
        pred = arr_pred[i]
        label = y[i]
        error = abs(label - pred)
        print(f"Actual: {label}     Prediction: {pred}      Error: {error}")

        if error < 0.5:
            print("CORRECT")
            correct += 1
        else:
            print("MISS")
            fault += 1

    print("##########################################")
    print(f"Correct: {correct}      Fault: {fault}")
    print(f"Accuracy: {correct / len(arr_pred)}")
    return (correct, fault, correct / len(arr_pred))


if __name__ == "__main__":
    name = ""

    if len(sys.argv) > 1:
        name = sys.argv[1]

    predict(name)
