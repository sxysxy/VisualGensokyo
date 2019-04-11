# coding=utf-8
# 命令行参数给出train的时候为训练模式，会搜索Convert.py中指定的分类，查找文件夹下xxxArrayData中的dataset.dat数据集，读取数据进行训练

#╮(╯▽╰)╭ 出于学习的一个项目，缺乏经验，代码写着写着就很乱了。

import os
import sys

TRAIN = len(sys.argv) > 1 and sys.argv[1] == "train"
import numpy as np
if not TRAIN:  #不是训练，禁用GPU
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
import tensorflow as tf
import pickle
import random
#从Convert里面导入一些东西
from Convert import extract_image
from Convert import BASE_DIR
from Convert import OUTPUT_SIZE
from Convert import CLASSES
from Convert import CLASSES_NAME

if not TRAIN:
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtMultimedia import *
    from PIL import Image
    import struct
    import PyQt5.sip

#返回一个矩阵Variable
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
#返回一个向量Variable
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

#-------------------------神经网络构建-----------------------------------------------------------------------
#神将网络的结构将在文档中说明
InputImg = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE, OUTPUT_SIZE, 3], name="PlaceHolder_InputImg") 
OutputAns = tf.placeholder(tf.float32, shape=[None, len(CLASSES)], name="PlaceHolder_OutputAns")
KeepProb = tf.placeholder(tf.float32, name="KeepProb")
TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

conv1 = tf.layers.conv2d(inputs=InputImg, filters=32, kernel_size=[5,5], padding="SAME", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="SAME", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

pool_flat = tf.reshape(pool2, [-1, int(OUTPUT_SIZE / 4) * int(OUTPUT_SIZE / 4) * 64])

dense1 = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu, 
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense1_dropout = tf.nn.dropout(dense1, KeepProb)

W_logistic = weight_variable([512, len(CLASSES)])
B_logistic = bias_variable([len(CLASSES)])
Predict = tf.nn.softmax(tf.matmul(dense1_dropout, W_logistic) + B_logistic)
#------------------------------------------------------------------------------------------------------------
Loss = -tf.reduce_sum(OutputAns * tf.log(tf.clip_by_value(Predict ,1e-6, 1e10)))  #描述误差
Train = tf.train.AdamOptimizer().minimize(Loss)     #使用AdamOptimizer优化器

if TRAIN:  #如果是训练模式，使用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0  
config.gpu_options.allow_growth = True     

#解出dir中的dataset.dat数据集
def extract_dataset(dir):
    datafile = os.path.join(dir, "dataset.dat")
    with open(datafile, "rb") as f:
        return pickle.load(f)
    return [[[[]]]]

saver = tf.train.Saver()  #用于储存或读取神经网络参数的saver

with tf.Session(config=config) as sess:
    #解包训练数据
    train_data = []
    test_data = []
    len_data = 0

    #读取所有训练数据
    def extract_all():
        global train_data, test_data, len_data
        for i in range(len(CLASSES)):
            dataset = extract_dataset(os.path.join(BASE_DIR, CLASSES[i] + "ArrayData"))
            label = [0.0] * len(CLASSES)
            label[i] = 1.0 
            datans = [label] * len(dataset)
            data = list(zip(dataset, datans))
            #np.random.shuffle(data)   #打乱
            '''
            l = len(data)
            p = int(l * 0.8)
            train_data += data[0:p]   #用于训练
            test_data += data[p:l]    #用于测试
            '''
            train_data += data
            del label
            del dataset
            del datans
            del data
        np.random.shuffle(train_data)
        len_data = len(train_data)
        test_data = train_data
    
    #sample_data 抽样，从train_data中抽样前num个训练数据，返回二元组作为输入答案和输出答案
    _sample_start = 0
    _sample_end = 0
    def sample_data(num): 
        global _sample_start, _sample_end
        _sample_end = _sample_start + num
        _dataset = []
        _datans = []
        if _sample_end >= len_data:
            for i in range(_sample_start, len_data):
                _dataset.append(train_data[i][0])
                _datans.append(train_data[i][1])
            _sample_start = 0
            _sample_end = _sample_end - len_data + 1
        for i in range(_sample_start, _sample_end):
            _dataset.append(train_data[i][0])
            _datans.append(train_data[i][1])
        _sample_start = _sample_end
        return _dataset, _datans

    def train_network():
        for _ in range(780):
            _dataset, _datans = sample_data(84)
            sess.run(Train, feed_dict={InputImg:_dataset, OutputAns:_datans, KeepProb:TRAIN_KEEP_PROB})
            del _dataset
            del _datans

    #测试准确度
    def test_acc():
        global test_data
        BATCH_SIZE = 100
        tot = len(test_data)
        _test_data = []
        _test_label = []
        for t in test_data:
            _test_data.append(t[0])
            _test_label.append(t[1])
        correct = 0
        for batch in range(int(tot / BATCH_SIZE)):
            batch_data = _test_data[batch*BATCH_SIZE:(batch*BATCH_SIZE+BATCH_SIZE)]
            batch_label = _test_label[batch*BATCH_SIZE:(batch*BATCH_SIZE+BATCH_SIZE)]
            correct += sess.run(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Predict, axis=1), tf.argmax(OutputAns, axis=1)), tf.float32)), 
                        feed_dict={InputImg:batch_data, OutputAns:batch_label, KeepProb:TEST_KEEP_PROB})
            del batch_data
            del batch_label
        _start = int(tot/BATCH_SIZE) * BATCH_SIZE
        _end = tot
        correct += sess.run(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Predict, axis=1), tf.argmax(OutputAns, axis=1)), tf.float32)), 
            feed_dict={InputImg:_test_data[_start:_end], OutputAns:_test_label[_start:_end], KeepProb:TEST_KEEP_PROB})
        del _test_data
        del _test_label
        return 1.0 * correct / tot

    if TRAIN:   #训练
        sess.run(tf.global_variables_initializer())
        print("读取数据ing...")
        extract_all()
        print("训练ing")
        train_network()
        saver.save(sess, "./trained_network")
        print("训练完毕，已保存神经网络参数")
        #test acc
        print("在原训练集上准确度为：%.2f%%" % (test_acc() * 100.0))
        del train_data
        del test_data
    else:    #仅测试
        print("读取已训练好的神经网络(./trained_network)")
        try:
            saver.restore(sess, "./trained_network") 
        except:
            print("读取神经网络参数./trained_network失败")
            sess.close()
            sys.exit(0)

        #图形化界面
        from VisualGensokyoUI import Ui_MainWindow

        class VisualGensokyoWindow(QMainWindow):
            def __init__(self, parent = None):
                super(VisualGensokyoWindow, self).__init__(parent)
                self.ui = Ui_MainWindow()
                self.ui.setupUi(self)
                self.ui.action_exit.triggered.connect(sys.exit)
                self.ui.action_open_img.triggered.connect(self.open_img)
                self.ui.action_introduce.triggered.connect(self.about_introduce)
                self.ui.action_about_VG.triggered.connect(self.about_VG)
                self.ui.action_get_src.triggered.connect(self.get_src)

                self.show()

            #打开一个选择文件的对话框选择图片文件
            def open_img(self):
                fd = QFileDialog(self)
                fd.setWindowTitle("选择一张图片")
                fd.setAcceptMode(QFileDialog.AcceptOpen)
                fd.setFileMode(QFileDialog.ExistingFile)
                fd.setViewMode(QFileDialog.Detail)
                
                if fd.exec() == QFileDialog.Accepted:
                    fn = fd.selectedFiles()[0]
                    self.predict_img(fn)

            def about_VG(self):
                text = '''  Visual Gensokyo是我(HfCloud, github账号sxysxy)大学一年级时学习深度学习技术，出于学习目的使用相关技术对东方Project角色图片进行分类做出的小程序。
3月底在QQ空间上展示了一下这个小程序的功能，没想到被广大车万厨热情转发（笑），于是厨力再次被激发，之后我继续收集图片数据，使用两块GTX1080Ti对神经网络进行了训练，才有了现在能辨
认20个角色的版本。\n版权说明:\n  *程序源代码以MIT协议(THE MIT LICENSE)发布，这意味着您可以自由地使用/修改/再次发布代码用于非商业与商业用途，但是应当保留我原作者的版权信息。
  *对于训练这个神经网络所使用的图片，我不具有版权，这些图片以及使用它们作成的训练数据集不会发布在程序中。但是我会提供下载链接，仅供训练使用。'''
                QMessageBox.about(self, "关于", text)
        
            def about_introduce(self):
                sep = ", "
                characters = sep.join(CLASSES_NAME)
                QMessageBox.about(self, "简要说明", "  Visual Gensokyo(视觉幻想乡)是一个使用卷积神经网络对东方Project角色图片进行辨识分类的小程序。目前认识的角色有:{}\n  对于可能性小于1%的类别不会显示在预测结果中，您可以通过查看程序的控制台得到对所有分类所预测的概率".format(characters))

            def get_src(self):
                QMessageBox.about(self, "获取源代码", "请前往 https://github.com/sxysxy/VisualGensokyo")

            #处理文件被拖拽到窗口
            def dragEnterEvent(self, eve):
                fn = eve.mimeData().text()
                fn = fn[8:len(fn)] #去掉开头的file:///
                eve.accept()
                self.predict_img(fn)

            def predict_img(self, fn):
                global sess, Predict
                img_data = extract_image(fn)
                if len(img_data) != OUTPUT_SIZE:
                    QMessageBox.warning(self, "错误", "无法作为图片解析选中的文件 {}".format(fn), QMessageBox.Ok)
                    return
                print("Predicting {}".format(fn))
                predict = list(zip(sess.run(tf.reduce_sum(Predict, axis=0), feed_dict={InputImg:[img_data], KeepProb:TEST_KEEP_PROB}), 
                                CLASSES_NAME))
                #按照概率从大到小排序
                predict.sort(reverse=True)
                print("完整预测结果:")
                print(predict)

                qimg = QImage(fn)
                if(qimg.isNull()):
                    #不知道为什么有时候用QImage打不开图片，于是使用PIL解析图片数据然后再传给QImage构造函数去创建图像

                    img = Image.open(fn, "r")
                    pix = img.load()
                    data = []
                    for i in range(img.size[1]):
                        for j in range(img.size[0]):
                            c = pix[j, i]        
                            data.append(c[0])
                            data.append(c[1])
                            data.append(c[2])
                            if(len(c) == 4):
                                data.append(c[3]) 
                            else:
                                data.append(255)  #补上Alpha通道
                    qimg = QImage(QByteArray(bytes(data)), img.size[0], img.size[1], QImage.Format_RGBA8888)
                    

                desktop = QApplication.desktop()
                #调整窗口大小，但是窗口大小不会最大超出屏幕的限制，最小不会小于100 x 100
                self.resize(max([min([qimg.width(), desktop.width()]), 100]), max(100, min([qimg.height(), desktop.height()])))
                #窗口居中显示
                self.move((desktop.width()-self.width()) / 2, (desktop.height()-self.height()) / 2) 
                                
                self.ui.display_area.resize(self.frameGeometry().width(), self.frameGeometry().height())
                qpixmap = QPixmap.fromImage(qimg)
                self.ui.display_area.setPixmap(qpixmap)

                res = "按照可能性排序:\n"
                for p in predict:
                    if p[0] >= 1e-2:
                        res += "%s %.2f%% \n" % (p[1], p[0]*100)

                QMessageBox.about(self, "预测结果", res)
                self.setWindowTitle("Visual Gensokyo - 当前预测最可能是：{}".format(predict[0][1]))

        try:
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #设置高dpi下自适应调整
        except:
            pass
        app = QApplication(sys.argv)
        app_window = VisualGensokyoWindow()
        sys.exit(app.exec())