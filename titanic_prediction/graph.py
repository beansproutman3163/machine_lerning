import os
import csv

import pandas
import matplotlib.pyplot as plt

import definition as Def

# main----------------------------------------------------------
if __name__ == "__main__":
    pass

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
    data_frame = (
        pandas.read_csv("data\\train.csv")
        .replace("male", Def.MALE)
        .replace("female", Def.FEMALE)
    )

    # Age が空白のものを抽出 (NaN != NaN)
    emptyAge = data_frame.query("Age != Age")
    print(emptyAge)

    # fill empty data [Age]
    data_frame["Age"].fillna(data_frame.Age.median(), inplace=True)

    splitData = []
    for survived in [Def.DEAD, Def.SURVIVED]:
        splitData.append(data_frame[data_frame.Survived == survived])
    plot_data = [i["Pclass"].dropna() for i in splitData]

    print(plot_data)
    axes = plt.subplot()
    axes.hist(
        plot_data,
        histtype="barstacked",
        bins=3,
        cumulative=False,
        rwidth=0.5,
        color=["red", "green"],
        label=["Dead", "Survived"],
    )
    axes.set_title("title")

    plt.show()
