import os
import csv

import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import definition as Def


if __name__ == "__main__":

    # replace male = 0, female = 1
    data_frame = (
        pandas.read_csv("data\\train.csv")
        .replace("male", Def.MALE)
        .replace("female", Def.FEMALE)
    )

    # fill empty data [Age]
    data_frame["Age"].fillna(data_frame.Age.median(), inplace=True)

    data_frame["FamliySIze"] = data_frame["SibSp"] + data_frame["Parch"] + 1
    data_frame_fmt = data_frame.drop(
        ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1
    )

    train_data = data_frame_fmt.values
    xs = train_data[:, 2:]
    y = train_data[:, 1]

    print("_________XS___________")
    print(xs)
    print("_________Y___________")
    print(y)
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(xs, y)

    test_df = (
        pandas.read_csv("data\\test.csv")
        .replace("male", Def.MALE)
        .replace("female", Def.FEMALE)
    )

    # 補完
    test_df["Age"].fillna(data_frame.Age.median(), inplace=True)
    test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
    test_df_fmt = test_df.drop(
        ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1
    )

    print(test_df_fmt)
    test_data = test_df_fmt.values
    xs_test = test_data[:, 1:]
    print(xs_test)
    output = forest.predict(xs_test)

    print(len(test_data[:, 0]), len(output))
    zip_data = zip(test_data[:, 0].astype(int), output.astype(int))
    predict_data = list(zip_data)

    import csv

    with open("data\\predict_result_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(test_data[:, 0].astype(int), output.astype(int)):
            writer.writerow([pid, survived])
