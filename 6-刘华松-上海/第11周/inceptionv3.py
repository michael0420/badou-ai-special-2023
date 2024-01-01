import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载预训练的InceptionV3模型
model = InceptionV3(weights='imagenet')

# 加载并预处理图像
img_path = 'lenna.png'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 使用模型进行预测
preds = model.predict(x)

# 解码预测结果
decoded_preds = decode_predictions(preds, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")