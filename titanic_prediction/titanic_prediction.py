import os
import pandas
import matplotlib.pyplot as PyPlot

import definition as Def

# change current dir
os.chdir(os.path.dirname(__file__))

#
# train.csv
# PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
#
# SibSp ... 兄弟、配偶者の数
# Parch ... 両親、子供の数
# Embarked ... 乗船した港 [Cherbourg, Queenstown, Southampton]

# replace male = 0, female = 1
dataFrame = pandas.read_csv("train.csv").replace("male", Def.MALE).replace("female", Def.FEMALE)

# Age が空白のものを抽出 (NaN != NaN)
emptyAge = dataFrame.query('Age != Age')
print(emptyAge)

# fill empty data [Age]
dataFrame["Age"].fillna(dataFrame.Age.median(), inplace=True)

splitData = []
for survived in [Def.DEAD, Def.SURVIVED]:
	splitData.append(dataFrame[dataFrame.Survived == survived])


figure = PyPlot.figure()


plotData = [i["Pclass"].dropna() for i in splitData]
PyPlot.hist(plotData, histtype="barstacked", bins=3, cumulative=False, rwidth=0.5, color=['red', 'green'], label=['Dead', 'Survived'])
PyPlot.show()


