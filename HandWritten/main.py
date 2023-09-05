import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] ="0"

# cpu=tf.config.list_physical_devices("CPU")
# tf.config.set_visible_devices(cpu)

np.random.seed(423)
batch_size = 128
epochs = 30
learning_rate = 0.001


mnist = tf.keras.datasets.mnist
(train_image, train_label), (test_image, test_label) = mnist.load_data()
x_train = train_image
x_test = test_image
y_train = train_label
y_test = test_label

# 数据类型转化成tf需要的
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 数据正则化，映射到(0,1)
x_train /= 255
x_test /= 255

# 数据维度转换，四维
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = models.Sequential([
    # 第一层卷积层，output 24*24
    layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1),use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # 第二层卷积层，output 20*20
    layers.Conv2D(filters=64, kernel_size=(5, 5), use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # 第三层卷积层，output 16*16
    layers.Conv2D(filters=96, kernel_size=(5, 5), use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # 第四层卷积层，output 12*12
    layers.Conv2D(filters=128, kernel_size=(5, 5), use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # 第五层卷积层，output 8*8
    layers.Conv2D(filters=160, kernel_size=(5, 5), use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Flatten(),
    # 全连接层
    layers.Dense(10),
    layers.BatchNormalization(),
    layers.Activation(tf.nn.softmax)

])
print(model.summary())

#学习率指数衰减
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=375, decay_rate=0.98)
#优化器
adam_optimizer = optimizers.Adam(learning_rate=lr_schedule)

# 编译模型
model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# 训练模型
train = model.fit(x=x_train, y=y_train, validation_split=0.2, shuffle=True,
                  batch_size=batch_size, epochs=epochs, verbose=2)


#绘制评估曲线
plt.plot(train.history['accuracy'], label='Accuracy')
plt.plot(train.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.75, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(train.history['loss'], label='Loss')
plt.plot(train.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 0.6])
plt.legend(loc='upper right')
plt.show()

#模型测试
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("test loss - ", round(test_loss, 4), " - test accuracy - ", round(test_acc, 4))


