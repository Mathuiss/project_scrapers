import pandas as pd
import numpy as np
import sys
from sklearn import model_selection


def load(name):
    # Loading the data
    ufc_data = pd.read_csv("UFC-data/ufc-fights-model.csv")
    ufc_data = ufc_data.drop(["Unnamed: 0"], axis=1)

    if name != "":
        select_fights = pd.concat([ufc_data.loc[ufc_data["Name"] == name], ufc_data.loc[ufc_data["Name.1"] == name]])
        ufc_data = select_fights

    return ufc_data


def hoe_encode(ufc_data):
    stance_open_0 = []
    stance_orthodox_0 = []
    stance_southpaw_0 = []
    stance_switch_0 = []

    stance_open_1 = []
    stance_orthodox_1 = []
    stance_southpaw_1 = []
    stance_switch_1 = []

    for i, row in ufc_data.iterrows():
        if row["Stance"] == "Open Sta":
            stance_open_0.append(1)
            stance_orthodox_0.append(0)
            stance_southpaw_0.append(0)
            stance_switch_0.append(0)
        elif row["Stance"] == "Orthodox":
            stance_open_0.append(0)
            stance_orthodox_0.append(1)
            stance_southpaw_0.append(0)
            stance_switch_0.append(0)
        elif row["Stance"] == "Southpaw":
            stance_open_0.append(0)
            stance_orthodox_0.append(0)
            stance_southpaw_0.append(1)
            stance_switch_0.append(0)
        elif row["Stance"] == "Switch":
            stance_open_0.append(0)
            stance_orthodox_0.append(0)
            stance_southpaw_0.append(0)
            stance_switch_0.append(1)

        if row["Stance.1"] == "Open Sta":
            stance_open_1.append(1)
            stance_orthodox_1.append(0)
            stance_southpaw_1.append(0)
            stance_switch_1.append(0)
        elif row["Stance.1"] == "Orthodox":
            stance_open_1.append(0)
            stance_orthodox_1.append(1)
            stance_southpaw_1.append(0)
            stance_switch_1.append(0)
        elif row["Stance.1"] == "Southpaw":
            stance_open_1.append(0)
            stance_orthodox_1.append(0)
            stance_southpaw_1.append(1)
            stance_switch_1.append(0)
        elif row["Stance.1"] == "Switch":
            stance_open_1.append(0)
            stance_orthodox_1.append(0)
            stance_southpaw_1.append(0)
            stance_switch_1.append(1)

    ufc_data["stance_open_0"] = stance_open_0
    ufc_data["stance_orthodox_0"] = stance_orthodox_0
    ufc_data["stance_southpaw_0"] = stance_southpaw_0
    ufc_data["stance_switch_0"] = stance_switch_0

    ufc_data["stance_open_1"] = stance_open_1
    ufc_data["stance_orthodox_1"] = stance_orthodox_1
    ufc_data["stance_southpaw_1"] = stance_southpaw_1
    ufc_data["stance_switch_1"] = stance_switch_1

    return ufc_data


def process(ufc_data, ratio):
    # Normalizing the data
    columns_to_normalize = ufc_data.columns
    columns_to_normalize = columns_to_normalize.drop(["Name", "Stance", "DOB", "Name.1", "Stance.1", "DOB.1", "Win"])
    ufc_data[columns_to_normalize] = ufc_data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    ufc_data[columns_to_normalize] = ufc_data[columns_to_normalize].fillna(0)
    x = ufc_data[columns_to_normalize].to_numpy()
    y = ufc_data["Win"].to_numpy().reshape((-1, 1))
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=ratio, random_state=2020)
    return (x_train, x_test, y_train, y_test)


def preprocess(name, test_ratio):
    data = load(name)
    data = hoe_encode(data)

    if len(data) == 1:
        return False

    minimum = 1 / len(data)
    maximum = 1 / len(data) * (len(data) - 1)

    if test_ratio < minimum:
        test_ratio = minimum
    elif test_ratio > maximum:
        test_ratio = maximum

    x_train, x_test, y_train, y_test = process(data, test_ratio)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    np.save(f"models/{name}x_train", x_train)
    np.save(f"models/{name}x_test",  x_test)
    np.save(f"models/{name}y_train", y_train)
    np.save(f"models/{name}y_test",  y_test)


if __name__ == "__main__":
    name = ""

    if len(sys.argv) > 1:
        name = sys.argv[1]

    data = load(name)
    data = hoe_encode(data)
    x_train, x_test, y_train, y_test = process(data, 0.25)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    np.save(f"models/{name}x_train", x_train)
    np.save(f"models/{name}x_test",  x_test)
    np.save(f"models/{name}y_train", y_train)
    np.save(f"models/{name}y_test",  y_test)
