# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import compare
from predict import predict
import train
from preprocessor import preprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
# First let's take a look at the data, that is available to us.
# The first file we have is called `ufc-fights-model.csv`.
# This file contains raw fight data, which we can use to train a model,
# to predict fights with.

df = pd.read_csv("UFC-data/ufc-fights-model.csv").drop(columns=["Unnamed: 0"])

print(df.columns)


# %%
# We can see that this dataset contains metrics for each fighter.
# Each fighter has a name, height, weight, reach, stance, date of birth,
# but also things like Take Down Average, Strength Defence, Strice Accuracy and Submission Average.
# This is the data we will use to train a model with, which will predict fights.
# The last column: `Win` will be used to label the data.
# Before the AI can train with this data, the data needs to be preprocessed.
# We can call the preprocessor script to preprocess the entire dataset, which will be used for training,
# or we can preprocess a single fighter, if we want to predict this fighters fights.


preprocess("Conor McGregor", 0.25)
preprocess("Khabib Nurmagomedov", 0.25)


# %%
preprocess("", 0.20)
train.main()


# %%
# This script has created 4 files, for each fighter, in the `models/` directory.
# This these 4 files are `x_train.npy`, `x_test.npy`, `y_train.npy` and `y_test.npy`.
# These are binary numpy files, which can directly be read with our model.
# Now that we have preprocessed Conor McGregor and Khabib Nurmagomedov, we can
# let our model predict those fights.


predict("Conor McGregor")
predict("Khabib Nurmagomedov")


# %%
# As you can see, Conor is much less predictable than Khabib.
# From the 9 fights the model predicted, it was right 33% of the time for Conor.
# For the 11 fights from Khabib however, the model was right on all 11 fights.
# This raises the question, if the model predicts fights purely based on skill,
# are there other factors that influence the outcome of a match?
# Even though the answer is almost certainly yes, we cannot draw the conclusion
# that it must be match fixing.
# Micha told us that the most likely reason for match fixing is making money on betting odds.
# This is why we compare the results of the AI with the odds of the bookkeepers.
# This can be done with the `compare.py` script, in which we compare the data we have of a
# fighter, with the odds the bookies gave.


c_results = compare.main("Conor McGregor")
k_results = compare.main("Khabib Nurmagomedov")


# %%
# This process has been run in the `batch_compare.py` script, for the entire dataset.
# All fights of each fighter have been analysed and stored in `analysis.csv`.
# We can load these and take a look at them.

df = pd.read_csv("analysis.csv")
remove = []

for i in range(len(df)):
    r = df.iloc[i]["red"]
    b = df.iloc[i]["blue"]

    remove_at = df[(df["red"] == r) & (df["blue"] == b)].index.max()

    if remove_at not in remove:
        remove.append(remove_at)

df = df.drop(remove)
df.to_csv("analysis2.csv")


# %%
# Looking at the data where the bookies outperform the AI.
idx = df[df["ai_err"] > df["bookies_err"]]

idx["delta"] = np.zeros((len(idx),))
idx.reset_index(drop=True, inplace=True)

for i in range(len(idx)):
    delta = abs(idx.iloc[i]["bookies"] - idx.iloc[i]["ai"])
    idx.at[i, "delta"] = delta

idx.to_csv("analysis3.csv")


# %%
# Filtering that data, looking for match where the AI is wrong.
df = pd.read_csv("analysis3.csv")

df = df[df["ai_err"] >= 0.5]
df = df.drop(columns=["Unnamed: 0"])
df.to_csv("analysis4.csv")

print(df.head())


# %%
# Looking for data where both the AI and the bookies are wrong.
df = pd.read_csv("analysis2.csv")

df = df[(df["ai_err"] >= 0.5) & (df["bookies_err"] >= 0.5)]
df = df.drop(columns=["Unnamed: 0"])
df.to_csv("analysis5.csv")

print(df.head())


# %%
# Looking at the fighters occurance in the last set.

fighters = pd.DataFrame(columns=["count"])

for i in range(len(df)):
    # f = ""
    # if df.iloc[i]["actual"] == 1:
    #     f = df.iloc[i]["blue"]
    # else:
    #     f = df.iloc[i]["red"]

    f = df.iloc[i]["red"]

    if f in fighters.index.values:
        fighters.at[f, "count"] += 1
    else:
        fighters.at[f, "count"] = 1

fighters = fighters.sort_values(by=["count"], ascending=False)
fighters.to_csv("analysis6.csv")

print(fighters.head(30))


# %%
# Where bookies get it wrong and ai gets it wrong, might indicate match fixing by outsiders.

# Where AI gets it wrong and bookies get it right, might indicate match fixing by bookies.

# In all cases the AI gets it wrong, it might indicate Error in the AI.

# In the sanddance graph where x = delta, and we color by ai err, and sort by ai, we can see that the green on the top is the outsiders match fixing, and the green on the bottom is the bookies match fixing, The area in the middle is where the AI was right, regardless of what the bookies thought.
# The further you go eft, the more certain you are of match fixing, the further you go to the right, the more certain you are of no match fixing.???
