# Project Scrapers

In this project we are researching match fixing in the UFC.

We are trying to gain a picture of what the chanses of match fixing in the UFC are. To accomplish this, we have 2 data sets. One data set contains information about fight metrics, and all sorts of skill based features. The other data set contains betting odds from book makers. Our goal is to create a skill-based model, capable of predicting the outcomes of matches, and comparing that to the betting odds of the bookmakers.

## Approach

The first thing we need to do is pre-process the data. This must be done because we need to train an AI, which can predict matches. Once preprocessing is done, we can start training the model. After the model is trained optimally, we can use the model to predict all fights of a certain fighter, or all fights of all fighters. We use the results from that to build a new dataset, called ```results.csv```, which can be found [here](https://github.com/Mathuiss/project_scrapers/blob/master/results.csv). These results can be used to track fighters that are predictable, or fighters that are unpredictable. The next step is to compare the predictions of the AI, with the predictions of the bookmakers. The problem with this however, is that not all fights are contained within both datasets. In the ```compare.py```, we will first extract all the fights that occur in both datasets, and then compare the results with bookmakers predictions. After we have done that we have an ```analysis.csv``` file. This dataset contains all the comparisons between each prediction, and the bookmakers odds. The final step is to use the ```dashboard.ipynb``` file to filter all usefull data, and search for match fixing, by looking for irregularities in the data. For data visualisation SandDance is used. This is a tool which can generate graohs and other visuals from ```.csv``` files.

## Pre-Processing

### Fight data

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

