# We now have the results.csv file, in which we can find the predictable and unpredictable fighters.
# We also have a compare script which can compare the fights of a signle fighter with the odds of
# the book keepers.
# We will

# Approach:
# Load the results.csv file
# We will load all results into a dataframe
# We will analyse and compare each fighter in the dataframe
# The results will be saved in the analysis.csv file

import pandas as pd
import compare

results = pd.read_csv("results.csv")

with open("analysis.csv", "w") as f:
    f.write("red,blue,actual,ai,ai_err,bookies,bookies_err\n")

    for i in range(len(results)):
        name = results.iloc[i]["name"]
        analysis = compare.main(name)

        for n in range(len(analysis)):
            r = analysis[n]["R"]
            b = analysis[n]["B"]
            actual = analysis[n]["ACTUAL"]
            ai = analysis[n]["AI"]
            ai_err = abs(actual - ai)
            bookies = analysis[n]["BOOKIES"]
            bookies_err = abs(actual - bookies)

            f.write(f"{r},{b},{actual},{ai},{ai_err},{bookies},{bookies_err}\n")
