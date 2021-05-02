# -*- coding: utf-8 -*-
# @Time : 2020/3/21 
# @File : inference.py
# @Software: PyCharm
import cv2
import os
import time

import numpy as np
import tensorflow as tf
from absl import flags, app
from absl.flags import FLAGS

from PIL import ImageFont, ImageDraw, Image
import winsound

from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms, pad_input_image, recover_pad_output, show_image
from network.network import SlimModel  # defined by tf.keras

flags.DEFINE_string('model_path', 'checkpoints/', 'config file path')

def parse_predict(predictions, priors, cfg):
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]
        score_idx = cls_scores > cfg['score_threshold']

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, cfg['nms_threshold'], cfg['max_number_keep'])

        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)

        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores

def sound():
    playsound("warning.mp3")


def main(_):
    global model
    cfg = config.cfg
    min_sizes = cfg['min_sizes']
    num_cell = [len(min_sizes[k]) for k in range(len(cfg['steps']))]

    try:
        model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)

        paths = [os.path.join(FLAGS.model_path, path)
                 for path in os.listdir(FLAGS.model_path)]
        latest = sorted(paths, key=os.path.getmtime)[-1]
        model.load_weights(latest)
        print(f"model path : {latest}")
        #new_input = tf.keras.Input(shape=(224,224,3))
        #x = model(new_input)
        #m = tf.keras.Model(inputs=new_input, outputs=x)
        model.save('final.h5')
        #converter = tf.lite.TFLiteConverter.from_keras_model(m)
        #tflite_model = converter.convert()
        #with open('model.tflite', 'wb') as f:
        #	f.write(tflite_model)
        # model.summary()
    except AttributeError as e:
        print('Please make sure there is at least one weights at {}'.format(FLAGS.model_path))

    capture = cv2.VideoCapture("http://192.168.0.8:4747/video")
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    FPS = 16

    priors, _ = priors_box(cfg, image_sizes=(480, 640))
    priors = tf.cast(priors, tf.float32)
    start = time.time()

    b,g,r,a = 0,0,255,0
    fontpath = "malgun.ttf"
    font = ImageFont.truetype(fontpath, 30)
    soundTime = time.time()

    while True:
        ret, frame = capture.read()
        if frame is None:
            print('No camera found')
        #print(frame.shape)

        current_time = time.time() - prev_time

        if (ret is True) and (current_time > 1.0/ FPS) :
            prev_time = time.time()

            h, w, _ = frame.shape
            img = np.float32(frame.copy())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img / 255.0 - 0.5

            predictions = model(img[np.newaxis, ...])
            boxes, classes, scores = parse_predict(predictions, priors, cfg)

            onmask=1

            if(len(classes) > 0):
                for i in classes:
                        if(i == 1):
                            onmask = 0
                        if(i == 2):
                            onmask = 1
                            break

            print(onmask)
                

            for prior_index in range(len(classes)):
                show_image(frame, boxes, classes, scores, h, w, prior_index, cfg['labels_list'])
            # calculate fps
            #fps_str = "FPS: %.2f" % (1 / (time.time() - start))
            start = time.time()

            warning = "마스크를 착용해주세요"
            if(onmask == 1 & len(classes) > 0):
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((25, 25),  "마스크를 착용해주세요", font=font, fill=(b,g,r,a))
                frame = np.array(img_pil)
                if(time.time() - soundTime > 5):
                    winsound.PlaySound("warning.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                    soundTime = time.time()
                #cv2.putText(frame, warning, (25, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                raise KeyboardInterrupt

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        app.run(main)
    except Exception as e:
        print(e)
        raise KeyboardInterrupt
