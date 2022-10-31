import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("CovidDeaths.csv")

data_ar = data.to_numpy()

total_deaths = [ x[6] for x in data_ar if(x[2]=="Russia" and x[8] > 0)]

date = np.arange(0, len(total_deaths))

plt.plot( date, total_deaths, lw=1)

plt.show()


