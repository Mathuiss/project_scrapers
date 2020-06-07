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


metrics = {"Fighters": [], "Accuracy": []}

for f in q_fighters:
    if preprocessor.preprocess(f, 0.99) == False:
        continue
    print(f"Preprocessed: {f}")
    correct, fault, accuracy = predict.predict(f)
    metrics["Fighters"].append(f)
    metrics["Accuracy"].append(accuracy)


with open("results.csv", "w") as f:
    f.write("name,accuracy,status")
for i in range(metrics["Fighters"]):
    if metrics["Accuracy"] < 0.736:
        f.write(f'{metrics["Fighters"][i]},{metrics["Accuracy"][i]},QUESTIONABLE')
        print(f'{metrics["Fighters"][i]}:    {metrics["Accuracy"][i]}       QUESTIONABLE')
    else:
        f.write(f'{metrics["Fighters"][i]},{metrics["Accuracy"][i]},GOOD')
        print(f'{metrics["Fighters"][i]}:    {metrics["Accuracy"][i]}       GOOD')
