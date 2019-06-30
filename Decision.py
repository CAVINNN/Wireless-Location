import numpy as np
import xlrd
from sklearn import tree


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


TrainData = xlrd.open_workbook('TrainingNew.xlsx')
table = TrainData.sheets()[0]
nrows = table.nrows
ncols = table.ncols
TrainX = [([0] * (ncols - 3)) for p in range(nrows - 1)]
TrainY = [([0] * 1) for p in range(nrows - 1)]
TrainCoor = [([0] * 2) for p in range(nrows - 1)]
for i in range(nrows - 1):
    TrainY[i][0] = table.cell(i + 1, 0).value
    for j in range(ncols - 3):
        TrainX[i][j] = table.cell(i + 1, j + 3).value
    for k in range(2):
        TrainCoor[i][k] = table.cell(i + 1, k + 1).value

X = np.array(TrainX)
Y = np.array(TrainY)

TestData = xlrd.open_workbook('Testing.xlsx')
testTable = TestData.sheets()[0]
testNrows = testTable.nrows
testNcols = testTable.ncols
TestX = [([0] * (testNcols - 3)) for p in range(testNrows - 1)]
TestCoor = [([0] * 2) for p in range(testNrows - 1)]
for i in range(testNrows - 1):
    for j in range(testNcols - 3):
        TestX[i][j] = testTable.cell(i + 1, j + 3).value
    for h in range(2):
        TestCoor[i][h] = testTable.cell(i + 1, h + 1).value
onlineX = np.array(TestX)
actualCoor = np.array(TestCoor)

for criterion in ('gini','entropy'):
    classifier = tree.DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
    classifier.fit(X, Y.ravel())
    y_pre = classifier.predict(onlineX)

    Error = []
    for i in range(testNrows - 1):
        curPre = int(y_pre[i])
        PredictCoor = [TrainCoor[curPre - 1][0], TrainCoor[curPre - 1][1]]
        Error.append(np.linalg.norm(PredictCoor - actualCoor[i, :]))
        # print(i, end=" ")
        # print(PredictCoor, end=" ")
        # print(actualCoor[i, :], end=" ")
        # print(np.linalg.norm(PredictCoor - actualCoor[i, :]))

    print("------------ DecisionTreeClassifier-", criterion, "--------")
    print("min Error:", min(Error))
    print("max Error:", max(Error))
    print("Average Error:", Get_Average(Error))

print()
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")

print()


for criterion in ('mse', 'friedman_mse', 'mae'):
    classifier = tree.DecisionTreeRegressor(criterion=criterion)
    classifier.fit(X, Y.ravel())

    y_pre = classifier.predict(onlineX)

    Error = []
    for i in range(testNrows - 1):
        curPre = int(y_pre[i])
        PredictCoor = [TrainCoor[curPre - 1][0], TrainCoor[curPre - 1][1]]
        Error.append(np.linalg.norm(PredictCoor - actualCoor[i, :]))
        # print(i, end=" ")
        # print(PredictCoor, end=" ")
        # print(actualCoor[i, :], end=" ")
        # print(np.linalg.norm(PredictCoor - actualCoor[i, :]))

    print("------------ DecisionTreeRegressor-", criterion, "--------")
    print("min Error:", min(Error))
    print("max Error:", max(Error))
    print("Average Error:", Get_Average(Error))
