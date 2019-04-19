# VisualGensokyo
Identifying pictures of Touhou project characters by Nerual Networks. 使用神经网络辨认东方project角色的图片

目前支持辨认这些角色的图片：博丽灵梦, 雾雨魔理沙, 古明地恋, 蕾米莉亚斯卡雷特, 十六夜咲夜, 琪露诺, 克劳恩皮丝, 芙兰朵露斯卡雷特, 帕秋莉诺雷姬, 西行寺幽幽子, 键山雏, 蓬莱山辉夜, 八云紫,风见幽香, 八意永琳, 多多良小伞, 魂魄妖梦, 东风谷早苗, 灵乌路空, 古名地觉

这是一个跨平台的程序，但是目前我只制作了windows上的发行版，可以直接双击可执行文件执行。
[发行版下载页面](https://github.com/sxysxy/VisualGensokyo/releases)

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
数据集:链接：https://pan.baidu.com/s/1y_mUeZmjd7pyHMJDvzT3JA 提取码：cavs 包括原图和作成的数据集

## 制作训练数据集
如果您有更多的角色的图片，可以使用Convert.py作成数据集。只需修改Convert.py中的CLASS和CLASS_NAME，加上新角色的名字。然后运行
```
python3 Convert.py
```
会重做全部的数据集，如果只想重做某一个角色的数据集，只需要
```
python3 Convert.py 角色名
```
角色名就是脚本文件目录下，存放这个角色的图片的文件夹的名字，例如Reimu，Koishi

## 已训练好的神经网络
在发行版的VisualGensokyo目录下，checkpoint和3个trained_network开头的文件就是保存下来的训练好的神经网络。使用tf.Saver就可以加载使用

## 版权声明
- 程序源代码以MIT协议(THE MIT LICENSE)发布，这意味着您可以自由地使用/修改/再次发布代码用于非商业与商业用途，但是应当保留我原作者的版权信息。

- 对于训练这个神经网络所使用的图片，我不具有版权，这些图片以及使用它们作成的训练数据集不会发布在程序中。但是我会提供下载链接，仅供训练使用。

# 示例：
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/1.png)
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/2.png)
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/3.png)
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/4.png)
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/5.png)

还有激动人心的幻视功能:

西红柿炒鸡蛋芙兰
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/6.png)
古名地三鲜
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/7.png)
![](https://github.com/sxysxy/VisualGensokyo/raw/master/Examples/8.png)