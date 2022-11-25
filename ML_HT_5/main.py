import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split



alldata = pd.read_csv("titanic.tsv", sep='\t')

alldata["Sex"] = alldata["Sex"].apply(
    lambda x: 0 if x in ["male"] else 1
)

alldata = alldata.rename(columns={"Sex" : "Is female"})

del alldata["Cabin"]
del alldata["Name"]

alldata = alldata[pd.notna(alldata["Age"])]
alldata = alldata[pd.notna(alldata["Embarked"])]

alldata = pd.get_dummies(alldata, columns=["Embarked"])

alldata["SOTON ticket"] = alldata["Ticket"].apply(
    lambda x: 1 if "SOTON" in x else 0
)

alldata["STON ticket"] = alldata["Ticket"].apply(
    lambda x: 1 if "STON" in x else 0
)

alldata["C.A. ticket"] = alldata["Ticket"].apply(
    lambda x: 1 if "C.A" in x else 0
)

alldata["PC ticket"] = alldata["Ticket"].apply(
    lambda x: 1 if "PC" in x else 0
)

alldata["Undefined ticket"] = alldata["Ticket"].apply(
    lambda x: 1 if x.isnumeric() else 0
)

alldata["Other tickets"] = alldata["Ticket"].apply(
    lambda x: 0 
        if "SOTON" in x or "STON" in x or "C.A." in x
        or "PC" in x or x.isnumeric()
        else 1
)

del alldata["Ticket"]

data = alldata

data = pd.read_csv("titanic_ans.csv")

FEATURES = ["Pclass", "Is female", "Age", "SibSp", "Parch", "Fare","Embarked_C","Embarked_Q","Embarked_S",]


data_train, data_test = train_test_split(data, test_size=0.2)


y_train = pd.DataFrame(data["Survived"])
x_train = pd.DataFrame(data[FEATURES])

model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())

y_expected = pd.DataFrame(data_test["Survived"])
x_test = pd.DataFrame(data_test[FEATURES])

y_predicted = model.predict(x_test)

precision, recall, fscore, support = precision_recall_fscore_support(
    y_expected, y_predicted, average="micro"
)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {fscore}")

score = model.score(x_test, y_expected)

print(f"Model score: {score}")