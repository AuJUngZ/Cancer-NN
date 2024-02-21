from NNFunction import *
import pandas as pd
import random

learning_rate = 0.9
desire_output = [1, 1]  # 1,1 is 2 0,0 is 4
error = 1

X = [5, 2, 1, 1, 2, 1.0, 3, 1, 1, 2]
W10 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
W11 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
W12 = [0.1, 0.2, 1]
W13 = [0.1, 0.2, 1]

# random weights except for the last one is 1
W10 = [random.uniform(-1, 1) for i in range(10)]
W11 = [random.uniform(-1, 1) for i in range(10)]
W12 = [random.uniform(-1, 1) for i in range(3)]
W13 = [random.uniform(-1, 1) for i in range(3)]
W10[9] = 1
W11[9] = 1
W12[2] = 1
W13[2] = 1

avg_error = 1
count_epoch = 0

while count_epoch < 50:
    avg_error = 0
    error = 0
    data = pd.read_csv("train.csv")
    for i in range(len(data)):
        X = data.iloc[i, 0:10].tolist()
        desire_output = [1, 1] if X[9] == 2 else [0, 0]
        X[9] = 1

        # Forward
        print("\n-------------------Forward------------------->")
        out10 = Nout(X, W10)
        y10 = sigmoid(out10)
        print("\nSum(V) of node 10 is: %8.3f, Y from node 10 is: %8.3f" % (out10, y10))

        out11 = Nout(X, W11)
        y11 = sigmoid(out11)
        print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (out11, y11))

        out12 = Nout([y10, y11, 1], W12)
        y12 = sigmoid(out12)
        print("\nSum(V) of node 12 is: %8.3f, Y from node 12 is: %8.3f" % (out12, y12))

        out13 = Nout([y10, y11, 1], W13)
        y13 = sigmoid(out13)
        print("\nSum(V) of node 13 is: %8.3f, Y from node 13 is: %8.3f" % (out13, y13))

        # Backward
        print("\n-------------------Backward------------------->")
        # Error of output node
        e12 = y12 - desire_output[0]
        e13 = y13 - desire_output[1]
        print("\nError of node 12 is: %8.3f, Error of node 13 is: %8.3f" % (e12, e13))
        avg_error += (e12 + e13) / 2
        print("\nError of network is: %8.3f" % error)

        # Node 12
        g12 = gradOut(e12, y12)
        del_w1012 = deltaw(learning_rate, g12, y10)
        del_w1112 = deltaw(learning_rate, g12, y11)
        del_bias12 = deltaw(learning_rate, g12, 1)
        W12 = [W12[0] + del_w1012, W12[1] + del_w1112, W12[2] + del_bias12]
        print("\nNew weights of node 12 are: %8.3f, %8.3f, %8.3f" % (W12[0], W12[1], W12[2]))

        # Node 13
        g13 = gradOut(e13, y13)
        del_w1013 = deltaw(learning_rate, g13, y10)
        del_w1113 = deltaw(learning_rate, g13, y11)
        del_bias13 = deltaw(learning_rate, g13, 1)
        W13 = [W13[0] + del_w1013, W13[1] + del_w1113, W13[2] + del_bias13]
        print("\nNew weights of node 13 are: %8.3f, %8.3f, %8.3f" % (W13[0], W13[1], W13[2]))

        # Node 10
        sumPrevNode10 = (W12[0] * g12) + (W13[0] * g13)
        g10 = gradH(y10, sumPrevNode10)
        del_w110 = deltaw(learning_rate, g10, X[0])
        del_w210 = deltaw(learning_rate, g10, X[1])
        del_w310 = deltaw(learning_rate, g10, X[2])
        del_w410 = deltaw(learning_rate, g10, X[3])
        del_w510 = deltaw(learning_rate, g10, X[4])
        del_w610 = deltaw(learning_rate, g10, X[5])
        del_w710 = deltaw(learning_rate, g10, X[6])
        del_w810 = deltaw(learning_rate, g10, X[7])
        del_w910 = deltaw(learning_rate, g10, X[8])
        del_bias10 = deltaw(learning_rate, g10, 1)
        W10 = [W10[0] + del_w110, W10[1] + del_w210, W10[2] + del_w310, W10[3] + del_w410, W10[4] + del_w510,
               W10[5] + del_w610, W10[6] + del_w710, W10[7] + del_w810, W10[8] + del_w910, W10[9] + del_bias10]
        print("\nNew weights of node 10 are: %8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f" % (
            W10[0], W10[1], W10[2], W10[3], W10[4], W10[5], W10[6], W10[7], W10[8], W10[9]))

        # Node 11
        sumPrevNode11 = (W12[1] * g12) + (W13[1] * g13)
        g11 = gradH(y11, sumPrevNode11)
        del_w111 = deltaw(learning_rate, g11, X[0])
        del_w211 = deltaw(learning_rate, g11, X[1])
        del_w311 = deltaw(learning_rate, g11, X[2])
        del_w411 = deltaw(learning_rate, g11, X[3])
        del_w511 = deltaw(learning_rate, g11, X[4])
        del_w611 = deltaw(learning_rate, g11, X[5])
        del_w711 = deltaw(learning_rate, g11, X[6])
        del_w811 = deltaw(learning_rate, g11, X[7])
        del_w911 = deltaw(learning_rate, g11, X[8])
        del_bias11 = deltaw(learning_rate, g11, 1)
        W11 = [W11[0] + del_w111, W11[1] + del_w211, W11[2] + del_w311, W11[3] + del_w411, W11[4] + del_w511,
               W11[5] + del_w611, W11[6] + del_w711, W11[7] + del_w811, W11[8] + del_w911, W11[9] + del_bias11]
        print("\nNew weights of node 10 are: %8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f" % (
            W11[0], W11[1], W11[2], W11[3], W11[4], W11[5], W11[6], W11[7], W11[8], W11[9]))
    avg_error /= len(data)
    count_epoch += 1

print("\n-------------------End of Training------------------->")
print("\nAverage Error: ", avg_error)
print("\nTotal Epochs: ", count_epoch)
print("\nFinal weights of node 10 are: ", W10)
print("\nFinal weights of node 11 are: ", W11)
print("\nFinal weights of node 12 are: ", W12)
print("\nFinal weights of node 13 are: ", W13)

# Testing
data = pd.read_csv("test.csv")
total = 0
correct = 0
for i in range(len(data)):
    X = data.iloc[i, 0:10].tolist()
    correct_ans = X[9]
    X[9] = 1
    # Forward
    print("\n-------------------Forward------------------->")
    out10 = Nout(X, W10)
    y10 = sigmoid(out10)

    out11 = Nout(X, W11)
    y11 = sigmoid(out11)

    out12 = Nout([y10, y11, 1], W12)
    y12 = sigmoid(out12)

    out13 = Nout([y10, y11, 1], W13)
    y13 = sigmoid(out13)

    y12 = 1 if y12 > 0.5 else 0
    y13 = 1 if y13 > 0.5 else 0
    result = [y12, y13]
    result = 2 if result == [1, 1] else 4
    print("\nDesire Output: ", [1, 1] if X[9] == 2 else [0, 0])
    print("\nOutput: ", [y12, y13])
    print("Correct Answer: ", correct_ans, "Predicted Answer: ", result)
    print("Correct" if correct_ans == result else "Wrong")
    if correct_ans == result:
        correct += 1
    total += 1
    print("\n------------------------------------------------->")

print("\n-------------------FINAL RESULT------------------->")
print("This test with model trained with ", count_epoch, " epochs with avg error: ", avg_error)
print("Total: ", total)
print("Correct: ", correct)
print("Wrong: ", total - correct)
print("Accuracy: ", correct / total * 100, "%")