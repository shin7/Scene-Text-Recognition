import os
import sys
import cv2
import math
import time
import torch
import random
import argparse
import numpy as np
import unicodedata as ud
from models import resnet50
import torch.nn.functional as F
from utils import np_to_variable
from utils import locality_aware_nms
from utils.ocr_util import print_seq_text
from utils.rbox_util import restore_rectangle
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

f = open('files/codec.txt', 'r')
codec = f.readlines()[0]
f.close()
print(len(codec))


def get_images(test_data_path):
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = locality_aware_nms.nms_locality(boxes.astype(np.float64), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def recognize(model, detection, img_data, input_img, debug=False):
    boxo = detection
    boxr = boxo[0:8].reshape(-1, 2)

    boxhelp = np.copy(boxr)
    boxr[0, :] = boxhelp[3, :]
    boxr[1, :] = boxhelp[0, :]
    boxr[2, :] = boxhelp[1, :]
    boxr[3, :] = boxhelp[2, :]

    center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4
    dw = boxr[2, :] - boxr[1, :]
    dh = boxr[1, :] - boxr[0, :]

    w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
    h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + random.randint(-2, 2)

    angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
    angle2 = math.atan2((boxr[3][1] - boxr[0][1]), boxr[3][0] - boxr[0][0])
    angle = (angle + angle2) / 2

    input_W = img_data.size(3)
    input_H = img_data.size(2)
    target_h = 44

    scale = target_h / h
    target_gw = int(w * scale + target_h / 4)
    target_gw = max(8, int(round(target_gw / 4)) * 4)
    xc = center[0]
    yc = center[1]
    w2 = w
    h2 = h

    # show pooled image in image layer
    scalex = (w2 + h2 / 4) / input_W
    scaley = h2 / input_H

    th11 = scalex * math.cos(angle)
    th12 = -math.sin(angle) * scaley
    th13 = (2 * xc - input_W - 1) / (input_W - 1)

    th21 = math.sin(angle) * scalex
    th22 = scaley * math.cos(angle)
    th23 = (2 * yc - input_H - 1) / (input_H - 1)

    t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
    t = torch.from_numpy(t).type(torch.FloatTensor)
    t = t.cuda()
    theta = t.view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
    x = F.grid_sample(img_data, grid)

    labels_pred = model.forward_ocr(x)

    if debug:
        x_d = x.data.cpu().numpy()[0]
        x_data_draw = x_d.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)

        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        x_data_draw = x_data_draw[:, :, ::-1]
        cv2.imshow('ocr_image', x_data_draw)

        cv2.imshow('img', input_img)

        cv2.waitKey(100)

    ctc_f = labels_pred.data.cpu().numpy()
    ctc_f = ctc_f.swapaxes(1, 2)
    labels = ctc_f.argmax(2)

    ind = np.unravel_index(labels, ctc_f.shape)
    conf = np.mean(np.exp(ctc_f[ind]))

    det_text, conf2, dec_s, splits = print_seq_text(labels[0, :], codec)

    return det_text, conf2, dec_s


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', nargs='?', type=str, default='../data/plot/train',
                        help='Path to test directory')
    parser.add_argument('--resume', nargs='?', type=str, default="checkpoints/LS1706203-400000.h5",
                        help='Path to previous saved model')
    parser.add_argument('--output_dir', nargs='?', type=str, default='outputs/test/', help='Path to output directory')
    parser.add_argument('--debug', nargs='?', type=bool, default=False, help='Debug')
    parser.add_argument('--save_img', nargs='?', type=bool, default=False, help='Save preview images')

    args = parser.parse_args()

    draw_font = ImageFont.truetype("files/arial-unicode-regular.ttf", 18)

    if args.save_img:
        save_img_path = args.output_dir + 'images/'
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    im_fn_list = get_images(args.test_data_path)

    model = resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (step {})".format(args.resume, checkpoint['step']))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()

    with torch.no_grad():
        for im_fn in im_fn_list:
            im = cv2.imread(im_fn)[:, :, ::-1]
            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=1280)
            images = np.asarray([im_resized], dtype=np.float)
            im_data = np_to_variable(images).permute(0, 3, 1, 2)

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            start = time.time()
            score, geometry = model(im_data)
            timer['net'] = time.time() - start

            score = score.data.cpu()[0].numpy()
            score = score.squeeze(0)

            geometry = geometry.data.cpu()[0].numpy()
            geometry = geometry.swapaxes(0, 1)
            geometry = geometry.swapaxes(1, 2)

            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
            print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

            boxes_out = np.copy(boxes)

            if boxes is not None:
                scores = boxes[:, 8].reshape(-1)
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            duration = time.time() - start_time
            print('[timing] {}'.format(duration))

            # save to file
            if boxes is not None:
                # print(os.path.basename(im_fn).split('.')[0].replace('ts', 'res'))
                res_file = os.path.join(args.output_dir,
                                        '{}.txt'.format(os.path.basename(im_fn).split('.')[0].replace('ts', 'res')))

                im_draw = np.copy(im)
                img_pil = Image.fromarray(im_draw)
                draw_img = ImageDraw.Draw(img_pil)

                with open(res_file, 'w') as f:
                    for bid, box in enumerate(boxes):
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue

                        det_text, conf, dec_s = recognize(model, boxes_out[bid], im_data, im[:, :, ::-1], args.debug)
                        print(det_text)

                        draw_text = det_text
                        try:
                            if len(det_text) > 0 and 'ARABIC' in ud.name(det_text[0]):
                                draw_text = det_text[::-1]
                        except:
                            pass

                        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:.2f},{9}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            scores[bid], draw_text
                        ))

                        width, height = draw_img.textsize(det_text, font=draw_font)
                        center = [box[0, 0] + 3, box[0, 1] - height - 2]

                        draw_img.text((center[0], center[1]), det_text, fill=(255, 0, 0), font=draw_font)
                        draw_img.polygon(
                            [box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]],
                            outline=(0, 255, 0))

                        im = np.asarray(img_pil)

            if args.save_img:
                img_path = os.path.join(args.output_dir + 'images/', os.path.basename(im_fn))
                cv2.imwrite(img_path, im[:, :, ::-1])
