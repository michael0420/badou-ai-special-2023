import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # kmeans 的流程：
    # 1、 有一组数据，
    # 2、然后设置扎堆数k，
    # 3、然后确定结束条件，如循环次数，还有聚类结果差异值小于x则终止
    # 4、 寻找聚合起始点
    # 5、确定结果后，重复校验x次
    # 所以 第一步
    # 获取图像高度、宽度
    H, W = gray.shape[:]
    # 改变图像数组，原图像是 512 * 512 的二维，现在变成一维数组
    data = gray.reshape((H * W, 1))
    data = np.float32(data)

    #第二步
    kmeansNum = 4

    #第三步
    '''
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
    其中，type有如下模式：
    —–cv2.TERM_CRITERIA_EPS: 精确度（误差）满足epsilon停止。
    —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
    —-cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX
    '''
    criteria = (cv2.TERM_CRITERIA_EPS,1,1)
    #第四步
    startpoint = cv2.KMEANS_RANDOM_CENTERS
    # 第五步，设置重复校验次数
    checkNum = 5
    compactness, labels, centers = cv2.kmeans(data,kmeansNum,None,criteria, checkNum ,startpoint)
    result = labels.reshape((H,W))
    result = np.array(result,np.float32)
    cv2.imshow("lenna聚类结果",result)
    cv2.waitKey()



