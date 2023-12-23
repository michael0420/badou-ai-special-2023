# 导入NumPy库并使用别名np
import numpy as np
# 导入自定义的utils模块
import utils
# 导入OpenCV库
import cv2
# 从Keras库中导入backend模块，并使用别名K
from keras import backend as K
# 从自定义的model模块中导入AlexNet类
from model.AlexNet import AlexNet

# 使用Keras的backend模块设置图像维度的顺序为'tf'（TensorFlow风格）
K.set_image_dim_ordering('tf')

# 如果当前脚本是主程序
if __name__ == "__main__":
    # 创建AlexNet模型的实例
    model = AlexNet()
    # 从指定路径加载预训练权重
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    # 读取测试图像（Test.jpg）
    img = cv2.imread("./Test.jpg")
    # 将图像从BGR颜色空间转换为RGB颜色空间
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像像素值归一化到0到1的范围
    img_nor = img_RGB / 255.0
    # 在第0维度上添加一个维度，以符合模型的输入要求
    img_nor = np.expand_dims(img_nor, axis=0)
    # 调用utils模块中的resize_image函数，将图像调整为指定大小（224x224）
    img_resize = utils.resize_image(img_nor, (224, 224))
    # 使用模型进行图像预测，找到最大概率对应的类别索引
    predicted_class_index = np.argmax(model.predict(img_resize))
    # 打印预测结果
    print(utils.print_answer(predicted_class_index))
    # 在窗口中显示原始图像
    cv2.imshow("original", img)
    # 等待用户按键，0表示无限等待
    cv2.waitKey(0)
