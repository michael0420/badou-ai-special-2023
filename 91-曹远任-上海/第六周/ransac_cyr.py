import numpy as np
import matplotlib.pyplot as plt
import math


def generatedata(datan, noisen):
    k = np.random.uniform(2, 15)
    b = np.random.uniform(0, 100)
    x = np.array(range(0, datan))
    noise = [np.random.normal(0, 150) for i in range(x.shape[0])]
    y = k * x + b + noise
    noisex = np.array([np.random.uniform(0, 400) for i in range(noisen)])
    noisey = np.array([np.random.uniform(-400, 400) for i in range(noisen)])
    return x, y, noisex, noisey, k, b


def distance(k, b, x, y):
    A = k
    B = -1
    C = b
    return abs(A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2)


def ransac(x, y):
    insetrate = 0.2
    tryk = 500
    maxerr = 50
    mininsetnum = 300
    bestinsetx = bestinsety = bestoutsetx = bestoutsety = np.array([])
    bestinsetnum = bestk = bestb = 0
    for ki in range(tryk):
        # 随机选取点作为内群
        setinorout = np.array([1 if np.random.random() < insetrate else 0 for i in range(x.shape[0])])
        insetx = insety = outsetx = outsety = np.array([])
        for i in range(setinorout.shape[0]):
            if setinorout[i] == 1:
                insetx = np.append(insetx, x[i])
                insety = np.append(insety, y[i])
            else:
                outsetx = np.append(outsetx, x[i])
                outsety = np.append(outsety, y[i])
        # 做最小二乘法
        k, b = np.polyfit(insetx, insety, 1)
        # 在“外群”里看有哪些点符合内群要求
        outtoinx = outtoiny = np.array([])
        for i in range(outsetx.shape[0]):
            if distance(k, b, outsetx[i], outsety[i]) <= maxerr:
                outtoinx = np.append(outtoinx, outsetx[i])
                outtoiny = np.append(outtoiny, outsety[i])
        # 删除外群对应元素
        for i in range(outtoinx.shape[0]):
            outsetx = np.delete(outsetx, outsetx == outtoinx[i])
            outsety = np.delete(outsety, outsety == outtoiny[i])
        # 添加内群对应元素
        insetx = np.concatenate((insetx, outtoinx))
        insety = np.concatenate((insety, outtoiny))
        if insetx.shape[0] >= mininsetnum and insetx.shape[0] > bestinsetnum:
            # 记录满足最少点且最优的内群数量，k，b，内外群点坐标
            bestinsetnum = insetx.shape[0]
            bestk = k
            bestb = b
            bestinsetx = insetx
            bestinsety = insety
            bestoutsetx = outsetx
            bestoutsety = outsety
    if bestinsetnum == 0:
        raise ValueError("没有满足最少内群点的模型，请重试")
    else:
        return bestk, bestb, bestinsetx, bestinsety, bestoutsetx, bestoutsety


totaln = 500
datan = 400
noisen = totaln - datan
# 生成数据
datax, datay, noisex, noisey, exactk, exactb = generatedata(datan, noisen)
x = np.concatenate((datax, noisex))
y = np.concatenate((datay, noisey))
# 做RANSAC
result = ransac(x, y)
# 画图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(result[2], result[3])
ax.scatter(result[4], result[5])
ax.plot(x, exactk * x + exactb)
ax.plot(x, result[0] * x + result[1])
k, b = np.polyfit(x, y, 1)
ax.plot(x, k * x + b)
plt.legend(["ransac data", "data", "exact system", "ransac system", "linar fit system"])
plt.show()
