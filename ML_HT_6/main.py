import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("data6.tsv", sep='\t')
data_y = pd.DataFrame(data['y'])


data_x = pd.DataFrame(data['x'])
data_x = data_x.sort_values
model_x_1 = LinearRegression()
model_x_1.fit(data_x, data_y.values.ravel())


plt.plot(data_x, model_x_1.predict(data_x), "ro")
plt.show()



poly_2 = PolynomialFeatures(2)
data_x_2 = poly_2.fit_transform(data_x)


model_x_2 = LinearRegression()


model_x_2.fit(data_x_2, data_y.values.ravel())


plt.plot(data_x, model_x_2.predict(data_x_2), "ro")
plt.show()

poly_5 = PolynomialFeatures(5)
data_x_5 = poly_5.fit_transform(data_x)
model_x_5 = LinearRegression()
model_x_5.fit(data_x_5, data_y.values.ravel())
plt.plot(data_x, model_x_5.predict(data_x_5), "ro")
plt.show()