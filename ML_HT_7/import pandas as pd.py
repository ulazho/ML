import pandas as pd

a = pd.DataFrame([1, 2, 3, 4, 5])
a = a.replace(2, 3)
print (a)