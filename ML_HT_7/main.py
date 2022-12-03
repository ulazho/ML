import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

data = pd.read_csv("communities.data")
data = data.replace("?", np.nan)
atributes = pd.read_csv("attributes.csv")

atributes = atributes.columns.to_list()



data_df = pd.DataFrame(data)
data_df.columns = atributes
data_df = data_df.sample(frac=1)

data_df = data_df.dropna(axis="columns")
atributes = data_df.columns


features = [x for x in atributes if x != "ViolentCrimesPerPop" and x != 'communityname']


data_train, data_test = train_test_split(data_df, test_size=0.2)



data_train_x = pd.DataFrame(data_train[features])
data_train_y = pd.DataFrame(data_train['ViolentCrimesPerPop'])
data_test_x = pd.DataFrame(data_test[features])
data_test_y = pd.DataFrame(data_test['ViolentCrimesPerPop'])

model = LinearRegression()
model.fit(data_train_x, data_train_y.values.ravel())



data_test_y_predicted = model.predict(data_test_x)

print(f"MSE without regularization {mean_squared_error(data_test_y, data_test_y_predicted)}")


min = 2
for x in np.arange(1, 7):

    clf = Ridge(alpha=x)
    clf.fit(data_train_x, data_train_y.values.ravel())
    clf_predicted = clf.predict(data_test_x)
    if(min > mean_squared_error(data_test_y, clf_predicted)):
        min = mean_squared_error(data_test_y, clf_predicted)
        min_count = x
print(f"MSE with regularization {min}")