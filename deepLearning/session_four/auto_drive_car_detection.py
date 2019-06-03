import argparse # 命令行参数解析,解析命令行输入参数
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from deepLearning.session_four.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from deepLearning.session_four.yad2k.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    # 检测是否含有对象
    box_scores = box_confidence * box_class_probs
    # 获取概率最高的对象是哪个
    box_classes = K.argmax(box_scores,axis=-1) # 按行返回最大值索引，axis=0按列
    box_classes_scores = K.max(box_scores,axis=-1) # 按行返回最大值，axis=0按列
    # 将anchor boxes中概率低于shreshold的对象剔除掉
    filtering_mask = box_classes_scores < threshold
    # 将不要的anchor boxes剔除掉
    scores = tf.boolean_mask(box_classes_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

# with tf.Session() as test_a:# TODO 答案不同
#     # 随机生成服从正态分布的数值，mean均值，stddev标准差
#     box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
#     boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
#     box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
#     scores, boxes, classes  = yolo_filter_boxes(box_confidence,boxes,box_class_probs, threshold=.4)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scors.shape = " + str(scores.shape))
#     print("boxes.shape = " + str(boxes.shape))
#     print("classes.shape = " + str(classes.shape))

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
# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4)
# print("iou = " + str(iou(box1, box2)))
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    # 最大值抑制，按照分数进行降序排序，对比留存下来的boxes的iou超过iou_threshold的删除
    nms_indices = tf.image.non_max_suppression(boxes, scores, iou_threshold=iou_threshold, max_output_size=max_boxes)
    # 根据索引获取相应的数值
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
# with tf.Session() as test_b:
#     scores = tf.random_normal([54,], mean=1, stddev=4, seed=1)
#     boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
#     classes = tf.random_normal([54,], mean=1, stddev=4, seed=1)
#     scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(boxes.eval().shape))
def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, max_boxes=max_boxes, iou_threshold=iou_threshold)

    return scores, boxes, classes
# with tf.Session() as test_b:
#     yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
#     scores, boxes, classes = yolo_eval(yolo_outputs)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))


# def predict(sess, image_file):
#
#     image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
#
#     out_scores, out_boxes, out_classes = None
#
#     print("Found {} boxes for {}".format(len(out_boxes), image_file))
#
#     colors = generate_colors(class_names)
#
#     draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#
#     image.save(os.path.join("out", image_file), qulity=90)
#
#     output_image = scipy.misc.imread(os.path.join("out", image_file))
#     imshow(output_image)
#
#     return out_scores, out_boxes, out_classes
# sess = K.get_session()
# class_names = read_classes("model_data/coco_classes.txt")
# anchors = read_anchors("model_data/yolo_anchors.txt")
# image_shape = (720., 1280.)
# yolo_model = load_model("model_data/yolo.h5")
# yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
# scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
# out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
