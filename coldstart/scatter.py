import numpy as np
# import matplotlib.pyplot as plt
import os  # 导入os库

import matplotlib.pyplot as plt

import Constants

plt.rc('font', family='Times New Roman')

def showScatter(x, y, title=''):
    label_font = {'family': 'Times New Roman',
                  'weight': '100',
                  'size': 25,
                  }

    fig = plt.figure(figsize=(6, 7), dpi=100)
    # plt.style.use('ggplot')
    # plt.title('Gowalla', font=label_font)  # 添加标题\n",

    plt.grid(color="#cccccc", which="both", linestyle='--', linewidth=1, zorder=0)

    # x = np.linspace(0, 10, 30)  # 产生0-10之间30个元素的等差数列
    # noise = np.random.randn(30)  # 产生30个标准正态分布的元素
    # y1 = x ** 2 + 2 * noise  # //产生叠加噪声的数据系列1
    # y2 = x ** 1 + 2 * noise  # 产生叠加噪声的数据系列2
    # y3 = x ** 1.5 + 2 * noise  # 产生叠加噪声的数据系列3\n"
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置字体为SimHei显示中文\n",
    # plt.rc('font', size=14)  # 设置图中字号大小\n",
    # plt.figure(figsize=(6, 4))  # 设置画布\n",
    plt.bar(x, y, width=1.1, color='#3B75AF', zorder=100)  # 绘制柱状图\n",
    plt.title(title, font=label_font) # 添加标题\n",
    plt.xlabel('The length of sequence', font=label_font)  # 添加横轴标签\n",
    plt.ylabel('The number of users', font=label_font)  # 添加y轴名称\n",
    plt.yticks(fontproperties='serif', size=25)
    plt.xticks(fontproperties='serif', size=25)
    # path = 'D:\\my_python\\ch3\\output\\'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # plt.savefig(path + 'scatter.jpg')  # 保存图片
    plt.savefig(Constants.DATASET+".png", bbox_inches='tight')  # 保存图片\n",
    plt.show()