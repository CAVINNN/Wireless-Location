import numpy as np
import xlrd
from sklearn import svm
from sklearn import ensemble
from sklearn import tree


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


# 训练集
TrainData = xlrd.open_workbook('TrainingNew.xlsx')
table = TrainData.sheets()[0]
nrows = table.nrows
ncols = table.ncols
TrainX = [([0] * (ncols - 3)) for p in range(nrows - 1)]  # 地点信号强度的list (二维数组， 95个list， 每个list中1501个路由器信号强度)
TrainY = [([0] * 1) for p in range(nrows - 1)]            # 唯一数据标识的list (二维数组， 95个list， 每个list中1个float数字)
TrainCoor = [([0] * 2) for p in range(nrows - 1)]         # 地点坐标的list (二维数组， 95个list， 每个list中1个坐标)
for i in range(nrows - 1):
    TrainY[i][0] = table.cell(i + 1, 0).value
    for j in range(ncols - 3):
        TrainX[i][j] = table.cell(i + 1, j + 3).value
    for k in range(2):
        TrainCoor[i][k] = table.cell(i + 1, k + 1).value

X = np.array(TrainX)  # 地点信号强度list，转换为95维数组对象
Y = np.array(TrainY)  # 唯一数据标识list，转换为95维数组对象

# 测试集
TestData = xlrd.open_workbook('Testing.xlsx')
testTable = TestData.sheets()[0]
testNrows = testTable.nrows
testNcols = testTable.ncols
TestX = [([0] * (testNcols - 3)) for p in range(testNrows - 1)]  # 地点信号强度的list (二维数组， 59个list， 每个list中1501个路由器信号强度)
TestCoor = [([0] * 2) for p in range(testNrows - 1)]             # 地点坐标的list (二维数组， 59个list， 每个list中1个坐标)
for i in range(testNrows - 1):
    for j in range(testNcols - 3):
        TestX[i][j] = testTable.cell(i + 1, j + 3).value
    for h in range(2):
        TestCoor[i][h] = testTable.cell(i + 1, h + 1).value
onlineX = np.array(TestX)        # 转换为59维数组对象
actualCoor = np.array(TestCoor)  # 转换为59维数组对象


classifier = svm.SVC(kernel='rbf', C=2.0, gamma='scale')
# classifier = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=10)
# classifier = tree.DecisionTreeClassifier(criterion='entropy')

classifier.fit(X, Y.ravel())  # 输入训练集的95维1501个信号强度数据，与相对应的95个类别。进行训练

y_pre = classifier.predict(onlineX)  # 输入测试集的59维1501个信号强度数据，得到每个维度相对应的类别

Error = []
for i in range(testNrows - 1):
    curPre = int(y_pre[i])
    PredictCoor = [TrainCoor[curPre - 1][0], TrainCoor[curPre - 1][1]]  # 第1个类别对应TrainCoor[0]的数据，以此类推
    Error.append(np.linalg.norm(PredictCoor - actualCoor[i, :]))  # 第i个维度中所有维度的数据（二维）
    print(i, end=" ")
    print(PredictCoor, end=" ")
    print(actualCoor[i, :], end=" ")
    print(np.linalg.norm(PredictCoor - actualCoor[i, :]))  # 求二范数：空间上两个向量矩阵的直线距离

print(Get_Average(Error))
