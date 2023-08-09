import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Functions
def check_quality(quality):
    if quality > 6.5:
        print("Quality is {} and it is \"Good Quality\"".format(quality))
    else:
        print("Quality is {} and it is \"Bad Quality\"".format(quality))

# Importing dataset
dataset = pd.read_csv("Datasets/winequality-red.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,train_size=0.75, random_state=3)

# Training the model
classifier = RandomForestClassifier(criterion='gini',random_state=1)
classifier.fit(x_train, y_train)

check_quality(classifier.predict([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]))