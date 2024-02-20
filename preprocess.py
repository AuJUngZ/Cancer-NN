import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

if __name__ == '__main__':
    # fetch dataset
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets
    # count unique values in target
    print(y.value_counts())
    # combine features and targets
    data = pd.concat([X, y], axis=1)

    # find missing values
    missing_values = data.isnull().sum()
    # Bare Nuclei has 16 missing values
    print(f'mode of Bare Nuclei: {data["Bare_nuclei"].mode()}')

    # fill missing values with mode
    data['Bare_nuclei'] = data['Bare_nuclei'].fillna(data['Bare_nuclei'].mode()[0])

    # slipt data into train and test
    Xtrain, Xtest, YTrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # combine features and targets for train and test
    train = pd.concat([Xtrain, YTrain], axis=1)
    test = pd.concat([Xtest, Ytest], axis=1)

    # save as csv
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)