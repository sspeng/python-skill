import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from keras import backend as K
import numpy as np
def iou(box1, box2):

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2-xi1, 0) * max(yi2-yi1, 0)

    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    iou_rate = inter_area / union_area

    return iou_rate
def draw_rectangle():
    """
    画矩形框
    :return:
    """

    fig = plt.figure() #创建图
    ax = fig.add_subplot(111) # 创建子图
    # ax = plt.gca() # 获得当前整张图表的坐标对象
    ax.invert_yaxis()  # y轴反向
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
    def add_rectangle(x1,y1,x2,y2,color="black"): # 输入剧性的对脚坐标
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color))
    add_rectangle(.2, .1, .4, .3)
    add_rectangle(.3, .1, .4, .3, color="red")
    add_rectangle(.3, .1, .4, .4, color="blue")
    add_rectangle(.1, .1, .4, .4, color="orange") #scores = np.array([.4,.5,.72,.9,.45],dtype=np.float32)
    add_rectangle(.1, .1, .4, .3, color="yellow")
    plt.show()
boxes = np.array([[.1,.2,.3,.4],[.1,.3,.3,.4],[.1,.3,.4,.4],[.1,.1,.4,.4],[.1,.1,.3,.4]], dtype=np.float32)
scores = np.array([.4,.5,.72,.9,.45],dtype=np.float32)
with tf.Session() as sess:
    selected_indices = sess.run(tf.image.non_max_suppression(boxes=boxes, scores=scores,iou_threshold=0.5, max_output_size=5))
    print(selected_indices)
    selected_boxes = sess.run(K.gather(boxes, selected_indices))
    print(selected_boxes)
draw_rectangle()
