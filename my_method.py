import random
import time
import datetime
import traceback
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil

def normal():
    # try:
    #     raise Exception("自定义错误异常")
    # except Exception as e:
    #     print(traceback.format_exc())
    timestamp = int(time.time())
    time_array = time.localtime(timestamp)  # 时间戳转时间数组（time.struct_time）
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_array)  # 时间戳转字符串
    time_array = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")  # 字符串转时间数组（time.struct_time）
    timestamp = time.mktime(time_array)  # 字符串转化为时间戳
    print(timestamp)
    now_time = datetime.datetime.now()  # datetime.datetime 类型
    timestamp = int(now_time.timestamp())  # datetime.datetime类型时间转化为时间戳
    str = ["%02d:%02d" % (int(i/2), (i % 2)*30) for i in range(24)]  # 字符串格式化
    e_dc_indate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 存入mysql数据库的格式
    # 排序
    homework_value = {"a": ["wer",5], "b": ["sdf",2], "c": ["vwd",7], "d": ["btr",1]}
    sort_list = sorted(homework_value.items(), key=lambda x: x[1][1], reverse=True)
    mysql = """
    UNIX_TIMESTAMP(e.gpsdate) 
    """

def singleton(cls):
    """
    装饰器实现单例模式
    :param cls: 实例：方法或者类
    :return:
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def use_matlab():
    """
    使用matplotlib画图
    :return:
    """
    def draw_line():
        """
        画折线图
        :return:
        """
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
        mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # plt.figure(figsize=(15, 15))  # 创建绘图对象
        plt.subplot(221)   # 构建2*2的方格，占据第一个位置
        x_value = range(24)
        y_value = [random.randint(0,10) for i in range(24)]
        plt.plot(x_value, y_value, marker='o')  # 在当前绘图对象进行绘图（两个参数是x,y轴的数据）,marker显示点
        # for a, b in zip(x_value, y_value):  # 增加点坐标数据
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
        plt.title("图像一")  # 标题
        plt.xlabel("hours")  # 横坐标标签
        plt.ylabel("total_num")  # 纵坐标标签
        # plt.savefig("time_feature.png")  # 保存图片
        plt.show()  # 显示图表
    def draw_rectangle():
        """
        画矩形框
        :return:
        """
        fig = plt.figure()  # 创建图
        ax = fig.add_subplot(111)  # 创建子图
        # ax = plt.gca() # 获得当前整张图表的坐标对象
        ax.invert_yaxis()  # y轴反向
        ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
        rect = plt.Rectangle((0.1, 0.1), 0.5, 0.3, fill=False) # 靠近原点的点坐标，长，宽
        ax.add_patch(rect)
        plt.show()

def data_to_csv():
    """
    导出csv、excel文件
    :return:
    """
    columns = ['size', 'age', 'height']  # 文件标题
    data = [[0, 1, 1], [2, 1, 2]]  # 文件内容
    file_data = pd.DataFrame(data, index=range(len(data)), columns=columns)
    file_data.to_excel("output.xls", index=False)
    # file_data.to_csv("output.csv", index=False, encoding="utf-8", sep="\t")
    print("导出成功")


def the_iterator():
    """
    迭代器
    :return:
    """
    def readInChunks(fileObj, chunkSize=1024*1024*100):
        while 1:
            data = fileObj.readlines(chunkSize)
            if not data:
                break
            yield "".join(data)
    def cut_file(path):
        f = open(path)
        export_path = "./data/totalExposureLog/"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        os.mkdir(export_path)
        for chuck in readInChunks(f):
            wrfile=open(export_path+"totalExposureLog_"+str(i)+".out",'w')
            wrfile.write(chuck)
            wrfile.close()
        f.close()

