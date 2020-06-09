import pandas as pd
import predict
import preprocessor
from subprocess import call

df_fights = pd.read_csv("UFC-data/ufc-fights-model.csv")
df_fights = df_fights.drop(["Unnamed: 0"], axis=1)
df_odds = pd.read_csv("UFC-data/fightodds.csv")
df_odds = df_odds.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

fighters = pd.concat([df_fights["Name"], df_fights["Name.1"]])
odds = pd.concat([df_odds["R_fighter"], df_odds["B_fighter"]])
q_fighters = []

for v in fighters.values:
    if v in odds.values and v not in q_fighters:
        q_fighters.append(v)


with open("results.csv", "w") as file:
    file.write("name,accuracy,correct,fault,status\n")

    for f in q_fighters:
        if preprocessor.preprocess(f, 0.99) == False:
            continue
        print(f"Preprocessed: {f}")
        correct, fault, accuracy = predict.predict(f)
        if accuracy > 0.736:
            file.write(f"{f},{accuracy},{correct},{fault},GOOD\n")
        else:
            file.write(f"{f},{accuracy},{correct},{fault},QUESTIONABLE\n")
