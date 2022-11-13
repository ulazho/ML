import pandas as pd
import numpy as np


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

print(alldata)