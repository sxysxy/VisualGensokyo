# VisualGensokyo
Identifying pictures of Touhou project characters by Nerual Networks. 使用神经网络辨认东方project角色的图片
这是一个跨平台的程序，但是目前我只制作了windows上的发行版，可以直接双击可执行文件执行(见Release page)

# 源代码使用说明：
## 安装所需的包
```
pip3 install pillow
pip3 install pyqt5
pip3 install tensorflow
```

## 运行
```
python3 VisualGensokyo.py
```

## 训练
下载训练数据集，解压到脚本文件目录
运行命令:
```
python3 VisualGensokyo.py train
```