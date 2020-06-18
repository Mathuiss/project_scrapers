import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout


def main():
    x_train = np.load("models/x_train.npy")
    x_test = np.load("models/x_test.npy")
    y_train = np.load("models/y_train.npy")
    y_test = np.load("models/y_test.npy")

    # Building the model
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(32,)))
    model.add(Dropout(0.8))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    # Compiling the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    arr_metrics = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
    model.summary()
    evaluation = model.evaluate(x_test, y_test)
    print("Validation Loss, Validation Accuracy")
    print(evaluation)

    ###################         name           #####################
    model.save("models/small_model_demo.h5")
    ###################         name           #####################

    print(arr_metrics.history.keys())
    plt.plot(arr_metrics.history["accuracy"])
    plt.plot(arr_metrics.history["val_accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"])
    plt.show()

    plt.plot(arr_metrics.history["loss"])
    plt.plot(arr_metrics.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Traing", "Val"])
    plt.show()


if __name__ == "__main__":
    main()
