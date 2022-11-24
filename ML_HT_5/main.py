import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

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