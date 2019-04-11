# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\VisualGensokyoUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(497, 334)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.display_area = QtWidgets.QLabel(self.centralwidget)
        self.display_area.setGeometry(QtCore.QRect(1, 4, 491, 301))
        self.display_area.setText("")
        self.display_area.setObjectName("display_area")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 497, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_open_img = QtWidgets.QAction(MainWindow)
        self.action_open_img.setObjectName("action_open_img")
        self.action_exit = QtWidgets.QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.action_about_VG = QtWidgets.QAction(MainWindow)
        self.action_about_VG.setObjectName("action_about_VG")
        self.action_get_src = QtWidgets.QAction(MainWindow)
        self.action_get_src.setObjectName("action_get_src")
        self.action_introduce = QtWidgets.QAction(MainWindow)
        self.action_introduce.setObjectName("action_introduce")
        self.menu.addAction(self.action_open_img)
        self.menu.addSeparator()
        self.menu.addAction(self.action_exit)
        self.menu_2.addAction(self.action_introduce)
        self.menu_2.addAction(self.action_about_VG)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.action_get_src)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VisualGensokyo"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "关于"))
        self.action_open_img.setText(_translate("MainWindow", "打开图片"))
        self.action_open_img.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.action_exit.setText(_translate("MainWindow", "退出"))
        self.action_exit.setShortcut(_translate("MainWindow", "Esc"))
        self.action_about_VG.setText(_translate("MainWindow", "关于Visual Gensokyo"))
        self.action_get_src.setText(_translate("MainWindow", "获取程序源代码"))
        self.action_introduce.setText(_translate("MainWindow", "简要说明"))


