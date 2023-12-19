import numpy as np
import cv2

def warpM(s,d):
    #防呆检查
    assert s.shape[0] == d.shape[0] and s.shape[0]>=4

    nums = s.shape[0]
    A = np.zeros((nums*2,8))
    b = np.zeros((nums*2,1))
    for i in range(nums):
        x,y=s[i]
        xx,yy =d[i]
        A[i*2,:] = [x,y,1,0,0,0,-x*xx,-y*xx]
        A[i*2+1,:] = [0,0,0,x,y,1,-x*yy,-y*yy]
        b[i*2] = xx
        b[i*2+1] = yy
    A = np.mat(A)

    w = A.I*b

    w = np.array(w).T[0]
    w = np.insert(w,8,values=1,axis=0)
    w = w.reshape((3,3))

    return w
if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = warpM(src, dst)
    print(warpMatrix)

    img = cv2.imread('photo1.jpg')
    ss = img.copy()
    cv2.imshow('img',img)

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    w0 = warpM(src,dst)

    ss = cv2.warpPerspective(ss, w0, (337, 488))

    cv2.imshow('ss',ss)

    cv2.waitKey(0)