# -*- coding: utf-8 -*-  
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
  
# VGGNet-16  
# 导入常用库，载入TensorFlow，参考：http://blog.csdn.net/Zhenguo_Yan/article/details/78226937
from datetime import datetime  
import math  
import time
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf  
  
# 创建卷积层并把本层的参数存入参数列表  
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):  
    n_in = input_op.get_shape()[-1].value  
  
    with tf.name_scope(name) as scope:  
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out],  
                                 dtype=tf.float32,  
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  
        # 对input_op进行卷积处理  
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')  
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)  
        biases = tf.Variable(bias_init_val, trainable=True, name='b')  
        z = tf.nn.bias_add(conv, biases)  
        activation = tf.nn.relu(z, name=scope)  
        #p += [kernel, biases]
        p += [kernel]  
        return activation  
  
# 定义全连接层创建函数fc_op  
def fc_op(input_op, name, n_out, p):  
    n_in = input_op.get_shape()[-1].value  
  
    with tf.name_scope(name) as scope:  
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out],  
                                 dtype=tf.float32,  
                                 initializer=tf.contrib.layers.xavier_initializer())  
        biases = tf.Variable(tf.constant(0.1, shape= [n_out],  
                                         dtype=tf.float32), name='b')  
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)  
        #p += [kernel, biases]  
        p += [kernel] 
        return activation  
  
# 定义最大池化层创建函数mpool_op  
def mpool_op(input_op, name, kh, kw, dh, dw):  
    return tf.nn.max_pool(input_op,  
                          ksize=[1, kh, kw, 1],  
                          strides=[1, dh, dw, 1],  
                          padding='SAME',  
                          name=name)  
  
# 创建VGGNet-16  
def inference_op(input_op, keep_prob):  
    p = []  

    # conv1  
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)  
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)  
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)  
  
    # conv2  
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)  
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)  
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)  
  
    # conv3  
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)  
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)  
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)  
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)  
  
    # conv4  
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=2512, dh=1, dw=1, p=p)  
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)  
  
    # conv5  
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)  
  
    # 将conv5输出结果扁平化  
    shp = pool5.get_shape()  
    flattened_shape = shp[1].value * shp[2].value * shp[3].value  
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")  
  
    # fc6  
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)  
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")  
  
    # fc7  
    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)  
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")  
  
    # fc8  
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1, p=p) 
    #二元分类用sigmoid, 多分类用softmax
    sigmoid = tf.nn.sigmoid(fc8)  
    predictions = tf.cast(sigmoid > 0.5, tf.int32)  
    return predictions, sigmoid, fc8, p  

def compute_cost(sigmoid, labels, p):
    '''
    计算损失函数
    参数：
    softmax: 归一化后的结果
    labels：真实的标签值
    p: 权重参数
    '''
    #print(labels)
    
    cross_entropy = -tf.reduce_mean(labels*tf.log(sigmoid) + (1-labels)*tf.log(1-sigmoid))
    for w in p:
        cross_entropy += 0.001*tf.nn.l2_loss(w) 
    loss = cross_entropy
    optimizer = tf.train.AdamOptimizer(2e-3).minimize(loss)
    predictions = tf.argmax(sigmoid, 1)
    correct_prediction = tf.equal(tf.cast(predictions, tf.int32), tf.cast(labels, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return optimizer, loss, accuracy


  
# 定义评测主函数  
def run_benchmark():
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 64,
                                                class_mode = 'binary')
    with tf.Graph().as_default():  
        image_size = 64
        #Define input data tensors  
        images = tf.placeholder(name='images', shape=[None,  
                                               image_size,  
                                               image_size, 3],  
                                              dtype=tf.float32)  
        labels = tf.placeholder(name='labels', dtype=tf.float32, shape=[None,1])
        keep_prob = tf.placeholder(name='keeprob', dtype=tf.float32)  

        predictions, sigmoid, fc8, p = inference_op(images, keep_prob)  
        optimizer, loss, accuracy = compute_cost(sigmoid, labels, p)
        #print(p)
        # 创建Session并初始化全局参数  
        init = tf.global_variables_initializer()  
        sess = tf.Session()  
        sess.run(init)  
        for i in range(epochs):
            for j in range(num_batches):
                batch_data, batch_labels = training_set.next()
                batch_labels = np.reshape(batch_labels, [-1,1])
                feed = {keep_prob: 0.5, images: batch_data, labels: batch_labels}
                _, l = sess.run([optimizer, loss], feed)
                if j % 30 == 0:
                    print('Loss:', round(l, 4))
        print('Testing Result.....')       
        batch_data, batch_labels = test_set.next()
        batch_labels = np.reshape(batch_labels, [-1,1])
        feed = {keep_prob: 1.0, images: batch_data, labels: batch_labels}
        acc = sess.run(accuracy, feed)
        print('Testing accuracy:', round(acc, 4))
        #time_tensorflow_run(sess, predictions, , "Forward")  
        #objective = tf.nn.l2_loss(fc8)  
        #grad = tf.gradients(objective, p)  
        #time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")  
  
if __name__ == '__main__':
    batch_size = 32
    num_batches = 500
    epochs = 100
    run_benchmark()  