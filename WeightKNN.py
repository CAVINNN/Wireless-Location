import numpy as np
import xlrd
from sklearn.neighbors import KNeighborsClassifier


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
TestY = [([0] * 1) for p in range(testNrows - 1)]
TestCoor = [([0] * 2) for p in range(testNrows - 1)]

for i in range(testNrows - 1):
    TestY[i][0] = testTable.cell(i + 1, 0).value
    for j in range(testNcols - 3):
        TestX[i][j] = testTable.cell(i + 1, j + 3).value
    for h in range(2):
        TestCoor[i][h] = testTable.cell(i + 1, h + 1).value

onlineX = np.array(TestX)
testingY = np.array(TestY)
actualCoor = np.array(TestCoor)

# NNnumber = 3
# weightPower = 4


for NNnumber in range(1, 10):
    for weightPower in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=NNnumber, p=2)
        knn.fit(X, Y.ravel())

        y_NN = knn.kneighbors(onlineX, n_neighbors=NNnumber)
        SimilarityNN = y_NN[0]
        IndexNN = y_NN[1]

        WeightedAvePredictCoor = [([0] * 2) for p in range(testNrows - 1)]

        for i in range(len(SimilarityNN)):
            x_Pre_pos = 0
            y_Pre_pos = 0
            totalWeight = 0
            for j in range(NNnumber):
                totalWeight = totalWeight + pow(1 / SimilarityNN[i][j], weightPower)

            for j in range(NNnumber):
                x_Pre_pos = x_Pre_pos + TrainCoor[IndexNN[i][j]][0] * (
                            pow(1 / SimilarityNN[i][j], weightPower) / totalWeight)
                y_Pre_pos = y_Pre_pos + TrainCoor[IndexNN[i][j]][1] * (
                            pow(1 / SimilarityNN[i][j], weightPower) / totalWeight)
                # print('Weight', i, ' ', j, ' ', pow(1 / SimilarityNN[i][j], weightPower) / totalWeight)
            WeightedAvePredictCoor[i][0] = x_Pre_pos
            WeightedAvePredictCoor[i][1] = y_Pre_pos

        ErrorWA = []
        for i in range(testNrows - 1):
            eachErrorWA = np.linalg.norm(WeightedAvePredictCoor[i] - actualCoor[i, :])
            # print(i, WeightedAvePredictCoor[i], eachErrorWA)
            ErrorWA.append(eachErrorWA)

        print('----------NNnumber:', NNnumber, ' weightPower:', weightPower, '---------------')
        print("min Error:", min(ErrorWA))
        print("max Error:", max(ErrorWA))
        print("Average Error:", Get_Average(ErrorWA))
