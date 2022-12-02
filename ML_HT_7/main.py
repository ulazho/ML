import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("communities.data")
atributes = pd.read_csv("attributes.csv")
data_df = pd.DataFrame(data)
data_df.replace('?' , np.nan)
data_df.columns = atributes.columns

data_df = data_df.sample(frac=1)
features = []

data_train, data_test = train_test_split(data_df, test_size=0.2)

print(data_df)