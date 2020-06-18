# Project Scrapers

In this project we are researching match fixing in the UFC.

We are trying to gain a picture of what the chanses of match fixing in the UFC are. To accomplish this, we have 2 data sets. One data set contains information about fight metrics, and all sorts of skill based features. The other data set contains betting odds from book makers. Our goal is to create a skill-based model, capable of predicting the outcomes of matches, and comparing that to the betting odds of the bookmakers.

## Approach

The first thing we need to do is pre-process the data. This must be done because we need to train an AI, which can predict matches. Once preprocessing is done, we can start training the model. After the model is trained optimally, we can use the model to predict all fights of a certain fighter, or all fights of all fighters. We use the results from that to build a new dataset, called ```results.csv```, which can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/results.csv). These results can be used to track fighters that are predictable, or fighters that are unpredictable. The next step is to compare the predictions of the AI, with the predictions of the bookmakers. The problem with this however, is that not all fights are contained within both datasets. In the ```compare.py```, we will first extract all the fights that occur in both datasets, and then compare the results with bookmakers predictions. After we have done that we have an ```analysis.csv``` file. This dataset contains all the comparisons between each prediction, and the bookmakers odds. The final step is to use the ```dashboard.ipynb``` file to filter all usefull data, and search for match fixing, by looking for irregularities in the data. For data visualisation SandDance is used. This is a tool which can generate graohs and other visuals from ```.csv``` files.


## Pre-Processing

### Fight Data

Before we can start to create a model, we need to take a thorough look at our data set. The file is called ```ufc-fights-model.csv``` and can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/UFC-data/ufc-fights-model.csv). This file contains fight metrics of 3139 fights. Most of the features can directly be normalized by scaling between 1 and 0. The column ```stance```, however, must be hot-one encoded. The last column ```Win```, contains our labels ```y```. The labels are already encoded ```1 or 0```, which is nice, since we can use binary crossentropy to classify fights.

### Preprocessor

The pre-processing work is done by the ```preprocessor.py``` script.
The implementation can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/preprocessor.py).

The script can be called on the command line via ```python preprocessor.py "Fighter Name"```, or if you want to preprocess all data, just omit the last argument; name.

Pre-processing consists of a few steps:
- Loading the data
- Hot-one encoding the data
- Normalizing the data
- Splitting the data into training and testing features and labels

Loading the data is done in the ```load()``` function. If a name is given, the script will load all fights where either the red or the blue fighter has that name, else it will simply load all fights. The reason preprocessing is done this way, is because after we have trained the model, we will want to predict a specific fighter at a time. We therefore must be able to pre-process data for that specific fighter.

Hot-one encoding the data is done by adding a new column for each stance in the ```stance``` column. This is done for each fighter, so ```4 stances x 2 fighters``` is a total of ```8``` new columns. After this we append a ```1``` to the current corresponding column, and a ```0``` to all other columns.

Normalizing the data is done by first stripping the columns that have no impact on the outcome of the fight what so ever, or features that have been incorporated in to data set in another maner. The ```Name, Date of Birth``` and the ```stance``` columns are dropped. We then proceed to apply the normalizing formula to each value in the data set, like so:

```python
ufc_data[columns_to_normalize] = ufc_data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
```

The last step is to split the data into training features, testing features, training labels and testing labels. Luckily ```sklearn.model_selection``` contains a function ```train_test_split()```, which can do this work for us. The function returns a tuple and takes the entire data set like so:

```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=ratio, random_state=2020)
```

The very last step is to use the ```numpy.save()``` method to save the preprocessed data sets to the hard drive, like so:

```python
np.save(f"models/{name}x_train", x_train)
np.save(f"models/{name}x_test",  x_test)
np.save(f"models/{name}y_train", y_train)
np.save(f"models/{name}y_test",  y_test)
```


## Training

The training of the model is done in the ```train.py``` script. The implementation can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/train.py).

The steps taken in this scripts are as follows:
- The training and testing data are loaded
- The model is defined and compiled
- The model is trained
- The training is evaluated

Loading the data from the hard drive is done with the ```numpy.load()``` method. The training data is saved in 4 different variables like so:

```python
x_train = np.load("models/x_train.npy")
x_test = np.load("models/x_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")
```

The next step if defining and compiling the model. This is done with the ```keras``` library. The model used in this project was of the type ```keras.model.Sequential()```. The model is defined like so:

```python
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
```

NOTE: This is not the model used in the data analysis. Even though this model performs really well, the ```small_model2.h5``` model performed even better.

Next, the model is trained and the results are stored in the ```arr_metrics``` variable. The model is also saved using the ```model.save()``` function. The last thing this script does, is evaluating the training by using ```matplotlib.pyplot``` to plot the ```history``` object in the ```arr_metrics``` variable. This will look something like this:

Accuracy
![Training Accuracy](https://cdn.discordapp.com/attachments/708243527389151254/723230779240611880/acc.svg)

Loss
![Training Loss](https://cdn.discordapp.com/attachments/708243527389151254/723230679269245039/loss.svg)


## Predicting

Predicting is done with the ```predict.py``` script. The implementation can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/predict.py)

The ```predict.py``` script takes an argument; name. This name can also be empty, in which case the script will predict all fighters within the preprocessed data set. The script will only load preprocessed data, so if you want to predict a fighter, you need to first preprocess that fighter. An example could be:

```bash
python preprocess.py "Conor McGregor"
python predict.py "Conor McGregor"
```

The script loads the model and the data, and then proceeds to call ```model.predict()``` on each data point. This will result in a prediction between ```0 and 1```. The script will also calculate error, the accuracy, the hits and the misses, like so:

```python
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
```


## Comparing

Comparing is done with the ```compare.py``` script. The implementation can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/compare.py).


### The Data

In essence we are comparing two data sets with eachother. We are comparing the results of the ```predict.py``` script with the odds of the bookmakers. We want to see if there is a difference between the models prediction, which is entirely skill-based, and the bookies predictions, which can be based on other factors as well, like match fixing. We will also be able to see if both the AI and the bookies get it wrong. This means that the problem is neither the skill of the fighter, or bookmakers bribing fighters to lose a certain match. The data set used in this experiment is ```fightodds.csv```, which can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/UFC-data/fightodds.csv).


### Compare

Comparing is a rather difficult process. We need to go through the matches of a fighter, and for each match, check if the match is also in the other data set. If that is not the case we drop the fight, else we can use it in our comparison. This is done by calling ```arr_compare(fights, odds)```. This function will loop through the fights, and call the ```present(pair, opposite)``` function like so:

```python
for i in range(len(fights)):
        if not present([fights.iloc[i]["R_fighter"], fights.iloc[i]["B_fighter"]], odds):
            drop_fights.append(i)

    for i in range(len(odds)):
        if not present([odds.iloc[i]["R_fighter"], odds.iloc[i]["B_fighter"]], fights):
            drop_odds.append(i)
```

NOTE: This is done for both fights and odds, since some fights are not in odds, but some odds are not in fights.

After making the lists equal, re-index the lists. This way we can iterate through one list and guarantee that we know where we all in the other list.

We then continue to call ```x = normalize(ufc_data)``` in order to create a normalized data set, which can be fet into a neural network. Normalizing the data is slightly different than in the ```preprocessor.py``` script, hence the seperate function. The normalize function in our case also calls ```fillna()```, and converts the data to a ```numpy.array()```, like so:

```python
columns_to_normalize = data.columns
columns_to_normalize = columns_to_normalize.drop(["R_fighter", "Stance", "DOB", "B_fighter", "Stance.1", "DOB.1", "Win"])
data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
data[columns_to_normalize] = data[columns_to_normalize].fillna(0)

return data[columns_to_normalize].to_numpy()
```

Lastly we call ```predict_compare()```, which will predict the match, and compare the results. This function is also slightly different from the ```predict.py``` script because it writes the data to the ```analysis.csv``` file, like so:

```python
def predict_compare(model, x, y, odds):
    results = []
    predictions = model.predict(x)
    predictions = np.array(predictions).reshape((-1, 1))

    for i in range(len(predictions)):
        pred = predictions[i, 0]
        r = odds.iloc[i]["R_fighter"]
        b = odds.iloc[i]["B_fighter"]
        odd = odds.iloc[i]["R_Implied_probability"]
        win = y.iloc[i]
        print(f"R: {r}, B: {b}, ACTUAL: {win}, AI: {pred}, BOOKIES: {odd}")
        results.append({"R": r, "B": b, "ACTUAL": win, "AI": pred, "BOOKIES": odd})

    return results
```


## Analysis

Analysis is done in the ```dashboard.ipynb``` file. The implementation can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/dashboard.ipynb).

### The Data

The data set, contained by ```analysis.csv``` is actually double. This is because we compare each fighter. If a fighter a fights against fighter b, we will get all matches for fighter a. If we later in the data set encounter fighter b, we will inevitably find the fight between fighter a and fighter b. This means that we have to clean up the data first.

### The Dashboard

The removing of the double items in the data set is done by a cell in the ```dashboard.ipynb``` file, like so:

```python
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

print(df.head())
```

If we use SandDance to plot the data set we get this image:

![DataSet](https://media.discordapp.net/attachments/708243527389151254/723102249299083294/unknown.png?width=757&height=677)

We can see that the AI is fairly certain about most fights, which are all in the left candle. All the candles right of the middle are wrong predictions. Of the predictions on the right candle, the AI was also fairly certain. Even though the AI has a %73,6 accuracy, there is still a significant candle of wrong predictions on the right. Contained within this candle are dark green spots. These are fights, which were wrongly predicted by both the AI and the bookmakers. Hence, these are fights we are interested in.


Something interesting happends when we only look at fights which wrongly predicted by the AI.

Match Fixing Propability
![Match Fixing Propability](https://media.discordapp.net/attachments/708243527389151254/723174530801205249/results_analysis.png?width=828&height=677)

Taking a look ath the above graph, we can see that there is an area in red-ish colors, where the AI was not that far off. The green spots represent fights which are very wrongly predicted by the AI. The x axis represents the error of the bookmakers. The more you go to the right, the more wrong the bookmakers get. Using this information we can draw ares, where we can estimate there is a lower chance of match-fixing, and where there is a higher chance of match-fixing.

The last thing the ```dashboard.ipynb``` file does, is it gives us a list of fighters, and how often they occur in fights that are wrongly predicted by both the AI and the bookmakers, like so:

```python
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
```

This gives us a list of names and counts, which can be used to further investigate the possibility of match fixing. This, ofcourse, has to be done in real-life, and cannot be done by the computer alone. Machine learning, however, is a powerfull tool to filter through large amounts of data, and build statistical models in a relatively easy fashion. At least for humans.

