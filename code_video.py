"""
基于视频的目标检测的测试和上面基于图像的目标检测原理都是一样的
不同的地方在于，需要从视频中提取每一帧进行目标检测
并将最终检测的结果形成视频保存
"""

import numpy as np
# import argparse
import imutils
import time
import cv2
import os



def NMSBoxes_fix(boxes, confidences, confidence_thre, nms_thre, class_id):
    class_id_set = set(class_id)  #  总共有几类
    result = []  #  用来存储结果的
    for cls in class_id_set:  # 遍历每个类别
        cls_boxes = []  # 用来保存每个类别的  边框
        cls_confidences = []  # 用来保存每个类别边框的分数
        indices = [i for i, c in enumerate(class_id) if c == cls] # 某一类在原始输入的所有索引
        for i in indices:
            cls_boxes.append(boxes[i])  # 把属于该类的框框和分数找出来
            cls_confidences.append(confidences[i])
        idxs = cv2.dnn.NMSBoxes(cls_boxes, cls_confidences, confidence_thre, nms_thre)  # 对每类进行 NMS 操作
        for i in idxs:  #  找出原始输入的索引，并把经过 NMS 操作后保留下来的框框的索引保存下来到一个列表中
            result.append([indices[i[0]]])  #
    return np.array(result)  #  opencv 原始的 NMS 输出是一个 np.array 的数据，所以我们也将其转化成指定格式



def vid_detect( pathIn='',
                pathOut=None,
                label_path='./cfg/coco.names',
                config_path='./cfg/yolov3_coco.cfg',
                weights_path='./cfg/yolov3_coco.weights',
                confidence_thre=0.5,
                nms_thre=0.30,
                jpg_quality=80):

    # 加载COCO数据集标签
    LABELS = open(label_path).read().strip().split("\n")


    # 获取颜色值
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 初始化VideoCapture类
    vc = cv2.VideoCapture(pathIn)
    writer = None
    (W, H) = (None, None)

    # 获取视频的总的帧数
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vc.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # 循环检测视频中的每一帧
    while True:
        # 读取帧
        # grabbed是一个bool，表示是否成功捕获帧
        # frame是捕获的帧
        (grabbed, frame) = vc.read()

        # 退出循环
        if not grabbed:
            break

        # 如果W,H为空，获取第一帧的width、height
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # 构造blob，并输入到网络中，执行Inference
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # 初始化列表，保存bbx信息
        boxes = []
        confidences = []
        classIDs = []

        # 循环每一个输出层的输出
        for output in layerOutputs:
            # 循环该层输出的每一个bbx
            for detection in output:
                # 获取当前bbx的信息
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # 类别最大概率与设定的阈值相比较
                if confidence > confidence_thre:
                    # bbx的坐标信息
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # 更新bbx列表
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # 执行NMS算法，去除重复的bbx
        # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
        idxs = NMSBoxes_fix(boxes, confidences, confidence_thre, nms_thre, classIDs)

        if len(idxs) > 0:
            # 循环提取每一个bbx坐标信息，使用OpenCV画在图上
            for i in idxs.flatten():
                # bbx坐标信息
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 画出bbx
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 检查writer是否为空
        if writer is None:
            # 初始化VideoWriteer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"MJPG")的"MJPG"需改成"mp4v"
            writer = cv2.VideoWriter(pathOut, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # 输出处理每一帧的时间，以及处理完视频总的时间
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

        # 写入当前帧
        writer.write(frame)

    # 释放文件指针
    print("[INFO] cleaning up...")
    writer.release()
    vc.release()



## 测试
src = './videos/'
dst = './output/'
vid_list = os.listdir(src)
for vid in vid_list:
    pathIn = src + vid
    pathOut = dst + vid
    vid_detect(pathIn, pathOut)
