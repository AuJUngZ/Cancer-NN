import pandas as pd

from NNFunction import *


def experimental(w10, w11, w12, w13, count_epoch, avg_error, is_train=False):
    if is_train:
        data = pd.read_csv("train.csv")
    else:
        data = pd.read_csv("test.csv")
    total = 0
    correct = 0
    for i in range(len(data)):
        X = data.iloc[i, 0:10].tolist()
        correct_ans = X[9]
        X[9] = 1
        # Forward
        # print("\n-------------------Forward------------------->")
        out10 = Nout(X, w10)
        y10 = sigmoid(out10)

        out11 = Nout(X, w11)
        y11 = sigmoid(out11)

        out12 = Nout([y10, y11, 1], w12)
        y12 = sigmoid(out12)

        out13 = Nout([y10, y11, 1], w13)
        y13 = sigmoid(out13)

        y12 = 1 if y12 > 0.5 else 0
        y13 = 1 if y13 > 0.5 else 0
        result = [y12, y13]
        result = 2 if result == [1, 1] else 4
        # print("\nDesire Output: ", [1, 1] if X[9] == 2 else [0, 0])
        # print("\nOutput: ", [y12, y13])
        # print("Correct Answer: ", correct_ans, "Predicted Answer: ", result)
        # print("Correct" if correct_ans == result else "Wrong")
        if correct_ans == result:
            correct += 1
        total += 1
        # print("\n------------------------------------------------->")

    print("\n-------------------FINAL RESULT------------------->")
    print("Train data" if is_train else "Test data")
    print("This test with model trained with ", count_epoch, " epochs with avg error: ", avg_error)
    print("Total: ", total)
    print("Correct: ", correct)
    print("Wrong: ", total - correct)
    print("Accuracy: ", correct / total * 100, "%")
    return correct / total * 100
