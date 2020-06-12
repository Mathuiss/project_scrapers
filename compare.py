import numpy as np
import pandas as pd
import preprocessor
import sys
from keras.models import load_model


# Normalize data for model input
def normalize(data):
    columns_to_normalize = data.columns
    columns_to_normalize = columns_to_normalize.drop(["R_fighter", "Stance", "DOB", "B_fighter", "Stance.1", "DOB.1", "Win"])
    data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    data[columns_to_normalize] = data[columns_to_normalize].fillna(0)
    return data[columns_to_normalize].to_numpy()


def present(pair, opposite):
    for i in range(len(opposite)):
        if pair[0] == opposite.iloc[i]["R_fighter"] and pair[1] == opposite.iloc[i]["B_fighter"]:
            return True

    return False


def arr_compare(fights, odds):
    drop_fights = []
    drop_odds = []

    print("Checking if fights in odds")
    for i in range(len(fights)):
        if not present([fights.iloc[i]["R_fighter"], fights.iloc[i]["B_fighter"]], odds):
            drop_fights.append(i)

    print("Checking if odds in fights")
    for i in range(len(odds)):
        if not present([odds.iloc[i]["R_fighter"], odds.iloc[i]["B_fighter"]], fights):
            drop_odds.append(i)

    if len(drop_fights) != 0:
        fights = fights.drop(drop_fights)

    if len(drop_odds) != 0:
        odds = odds.drop(drop_odds)

    return (fights, odds)


def predict_compare(model, x, y, odds):
    results = []
    predictions = model.predict(x)
    predictions = predictions.reshape((-1, 1))

    for i in range(len(predictions)):
        pred = predictions[i]
        odd = odds.iloc[i]["R_Implied_probability"]
        r = odds.iloc[i]["R_fighter"]
        b = odds.iloc[i]["B_fighter"]
        win = y.iloc[i]
        print(f"R: {r}, B: {b}, ACTUAL: {win}, AI: {pred}, BOOKIES: {odd}")
        results.append({"R": r, "B": b, "ACTUAL": win, "AI": pred, "BOOKIES": odd})

    return results


def main(name):
    # Load actual odds
    odds = pd.read_csv("UFC-data/fightodds.csv")

    # Load fighter odds
    fights_as_r = odds.loc[odds["R_fighter"] == name].drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    fights_as_b = odds.loc[odds["B_fighter"] == name].drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    fight_odds = pd.concat([fights_as_r, fights_as_b], ignore_index=True)

    # Load all fight metrics for that fighter
    ufc_data = preprocessor.hoe_encode(preprocessor.load(name))

    ufc_data = ufc_data.sort_values(by=["Name", "Name.1"])
    fight_odds = fight_odds.sort_values(by=["R_fighter", "B_fighter"])

    ufc_data.reset_index(drop=True, inplace=True)
    fight_odds.reset_index(drop=True, inplace=True)

    ufc_data = ufc_data.rename(columns={"Name": "R_fighter", "Name.1": "B_fighter"})
    ufc_data, fight_odds = arr_compare(ufc_data, fight_odds)

    ufc_data.reset_index(drop=True, inplace=True)
    fight_odds.reset_index(drop=True, inplace=True)

    x = normalize(ufc_data)
    return predict_compare(load_model("models/small_model4.h5"), x, ufc_data["Win"], fight_odds)


if __name__ == "__main__":
    name = sys.argv[1]
    main(name)
