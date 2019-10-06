import os
import csv

import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
data_frame = pandas.read_csv("train.csv").replace("male", Def.MALE).replace("female", Def.FEMALE)

# Age が空白のものを抽出 (NaN != NaN)
emptyAge = data_frame.query('Age != Age')
print(emptyAge)

# fill empty data [Age]
data_frame["Age"].fillna(data_frame.Age.median(), inplace=True)

splitData = []
for survived in [Def.DEAD, Def.SURVIVED]:
	splitData.append(data_frame[data_frame.Survived == survived])
plot_data = [i["Pclass"].dropna() for i in splitData]

print(plot_data)
axes = plt.subplot()
axes.hist(plot_data, histtype="barstacked", bins=3, cumulative=False, rwidth=0.5, color=['red', 'green'], label=['Dead', 'Survived'])
axes.set_title('title')

#plt.show()


data_frame["FamliySIze"] = data_frame["SibSp"] + data_frame["Parch"] + 1
data_frame_fmt = data_frame.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

train_data = data_frame_fmt.values
xs = train_data[:, 2:]
y = train_data[:, 1]

print("_________XS___________")
print(xs)
print("_________Y___________")
print(y)
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(xs, y)

test_df = pandas.read_csv("test.csv").replace("male", Def.MALE).replace("female", Def.FEMALE)

# 補完
test_df["Age"].fillna(data_frame.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df_fmt = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)


print(test_df_fmt)
test_data = test_df_fmt.values
xs_test = test_data[:, 1:]
print(xs_test)
output = forest.predict(xs_test)

print(len(test_data[:, 0]), len(output))
zip_data = zip(test_data[:, 0].astype(int), output.astype(int))
predict_data = list(zip_data)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])

