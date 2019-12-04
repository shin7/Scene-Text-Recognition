import os
import sys
import cv2
import math
import torch
import losses
import random
import argparse
import numpy as np
import unicodedata as ud
from models import resnet50
from datasets import dataset
import torch.nn.functional as F
from datasets import ocr_dataset
from utils import np_to_variable
from torch_baidu_ctc import CTCLoss
from utils.ocr_util import print_seq_text
from utils.rbox_util import draw_box_points

f = open('files/codec.txt', 'r')
codec = f.readlines()[0]
codec_rev = {}
index = 4
for i in range(0, len(codec)):
    codec_rev[codec[i]] = index
    index += 1
f.close()

batch_per_epoch = 5000
print_interval = 100


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.h5'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def recognizer(images, img_data, gt_output, label_output, model, ctc_loss, norm_height, debug=False):
    ctc_loss_count = 0
    loss = torch.from_numpy(np.asarray([0])).type(torch.FloatTensor).cuda()

    for bid in range(len(gt_output)):

        gts = gt_output[bid]
        lbs = label_output[bid]

        gt_proc = 0
        gt_good = 0

        if debug:
            img_draw = images[bid]
            img_draw = img_draw[:, :, ::-1]
            img_draw = np.asarray(img_draw, dtype=np.uint8)

        for gt_id in range(0, len(gts)):
            gt = gts[gt_id]
            gt_txt = lbs[gt_id]

            if gt_txt.startswith('#'):
                continue

            if gt[:, 0].max() > img_data.size(3) or gt[:, 1].max() > img_data.size(3):
                continue

            if gt.min() < 0:
                continue

            center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4
            dw = gt[2, :] - gt[1, :]
            dh = gt[1, :] - gt[0, :]

            w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
            h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + random.randint(-2, 2)

            if h < 8:
                # print('too small h!')
                continue

            angle = math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0])
            angle2 = math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0])
            angle_gt = (angle + angle2) / 2

            input_W = img_data.size(3)
            input_H = img_data.size(2)
            target_h = norm_height

            scale = target_h / h
            target_gw = int(w * scale) + random.randint(0, int(target_h))
            target_gw = max(8, int(round(target_gw / 4)) * 4)

            xc = center[0]
            yc = center[1]
            w2 = w
            h2 = h

            # show pooled image in image layer
            scalex = (w2 + random.randint(0, int(h2))) / input_W
            scaley = h2 / input_H

            th11 = scalex * math.cos(angle_gt)
            th12 = -math.sin(angle_gt) * scaley
            th13 = (2 * xc - input_W - 1) / (input_W - 1)

            th21 = math.sin(angle_gt) * scalex
            th22 = scaley * math.cos(angle_gt)
            th23 = (2 * yc - input_H - 1) / (input_H - 1)

            t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
            t = torch.from_numpy(t).type(torch.FloatTensor)
            t = t.cuda()
            theta = t.view(-1, 2, 3)

            grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
            x = F.grid_sample(img_data[bid].unsqueeze(0), grid)

            gt_labels = [codec_rev[' ']]
            for k in range(len(gt_txt)):
                if gt_txt[k] in codec_rev:
                    gt_labels.append(codec_rev[gt_txt[k]])
                else:
                    # print('Unknown char: {0}'.format(gt_txt[k]))
                    gt_labels.append(3)
            gt_labels.append(codec_rev[' '])

            if 'ARABIC' in ud.name(gt_txt[0]):
                gt_labels = gt_labels[::-1]

            labels_pred = model.forward_ocr(x)

            label_length = [len(gt_labels)]
            probs_sizes = torch.IntTensor(
                [(labels_pred.permute(2, 0, 1).size()[0])] * (labels_pred.permute(2, 0, 1).size()[1]))
            label_sizes = torch.IntTensor(torch.from_numpy(np.array(label_length)).int())
            labels = torch.IntTensor(torch.from_numpy(np.array(gt_labels)).int())

            loss = loss + ctc_loss(labels_pred.permute(2, 0, 1), labels, probs_sizes, label_sizes).cuda()
            ctc_loss_count += 1

            if debug:
                x_d = x.data.cpu().numpy()[0]
                x_data_draw = x_d.swapaxes(0, 2)
                x_data_draw = x_data_draw.swapaxes(0, 1)

                x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
                x_data_draw = x_data_draw[:, :, ::-1]
                cv2.imshow('img_ocr', x_data_draw)

                draw_box_points(img_draw, gt, color=(0, 255, 0))
                cv2.imshow('img', img_draw)
                cv2.waitKey(100)

            gt_proc += 1

            ctc_f = labels_pred.data.cpu().numpy()
            ctc_f = ctc_f.swapaxes(1, 2)
            labels = ctc_f.argmax(2)
            det_text, conf, dec_s, splits = print_seq_text(labels[0, :], codec)
            if debug:
                print('{0} \t {1}'.format(det_text, gt_txt))
            if det_text.lower() == gt_txt.lower():
                gt_good += 1

            # if ctc_loss_count > 128 or debug:
            #     break

    if ctc_loss_count > 0:
        loss /= ctc_loss_count

    return loss, gt_good, gt_proc


def main(args):
    if args.checkpoint == '':
        args.checkpoint = 'checkpoints'
    print(('checkpoint path: %s' % args.checkpoint))
    print(('init lr: %.8f' % args.lr))
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    start_step = 0

    train_loader = dataset.get_batch(num_workers=args.num_workers, input_dirs=args.input_dirs,
                                     input_size=args.input_size, batch_size=args.batch_size, vis=args.debug)

    # Load OCR dataset
    ocr_loader = ocr_dataset.get_batch(num_workers=2, input_list=args.ocr_input_list,
                                       batch_size=args.ocr_batch_size, norm_height=args.norm_height)

    model = resnet50(pretrained=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('Training from scratch.')

    model.train()

    ctc_loss = CTCLoss()

    train_loss_val, counter = 0, 0
    ctc_loss_val = 0
    good_all = 0
    gt_all = 0

    for step in range(start_step, args.max_iterators):
        # Localization data loader
        images_org, score_maps, geo_maps, training_masks, gt_outputs, label_outputs = next(train_loader)
        images = np_to_variable(images_org).permute(0, 3, 1, 2)
        score_maps = np_to_variable(score_maps).permute(0, 3, 1, 2)
        training_masks = np_to_variable(training_masks).permute(0, 3, 1, 2)
        geo_maps = np_to_variable(geo_maps)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Train the network localization and recognition
        score_pred, geo_pred = model(images)

        # Calculating the Loss
        loss = losses.loss(score_maps, score_pred, geo_maps, geo_pred, training_masks)

        train_loss_val += loss.item()

        try:
            if step > 10000:
                recog_loss, gt_b_good, gt_b_all = recognizer(images_org, images, gt_outputs, label_outputs, model,
                                                             ctc_loss, args.norm_height, args.ocr_debug)
                ctc_loss_val += recog_loss.item()
                loss = loss + recog_loss
                gt_all += gt_b_all
                good_all += gt_b_good
        except Exception:
            pass

        # Recognition data loader
        ocr_images, labels, labels_length = next(ocr_loader)
        ocr_images = np_to_variable(ocr_images).permute(0, 3, 1, 2)
        labels_pred = model.forward_ocr(ocr_images)

        probs_sizes = torch.IntTensor(
            [(labels_pred.permute(2, 0, 1).size()[0])] * (labels_pred.permute(2, 0, 1).size()[1]))
        label_sizes = torch.IntTensor(torch.from_numpy(np.array(labels_length)).int())
        labels = torch.IntTensor(torch.from_numpy(np.array(labels)).int())
        loss_ocr = ctc_loss(labels_pred.permute(2, 0, 1), labels, probs_sizes, label_sizes) / ocr_images.size(0) * 0.5

        # Calculating the Gradients
        loss_ocr.backward()
        loss.backward()

        # Update the weights
        optimizer.step()

        counter += 1

        if (step + 1) % print_interval == 0:
            train_loss_val /= counter
            ctc_loss_val /= counter

            print('\nEpoch: %d[%d] | LR: %f | Loss: %.3f | CTC_Loss: %.3f | Rec: %.5f' % (
                (step + 1) / batch_per_epoch, step + 1, optimizer.param_groups[0]['lr'], train_loss_val, ctc_loss_val,
                good_all / max(1, gt_all)))

            train_loss_val, counter = 0, 0
            ctc_loss_val = 0
            good_all = 0
            gt_all = 0

        if (step + 1) % batch_per_epoch == 0:
            checkpoint_file_name = 'LS1706203-{}.h5'.format(step + 1)
            save_checkpoint({
                'step': step + 1,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer': optimizer.state_dict(),
            }, checkpoint=args.checkpoint, filename=checkpoint_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--input_dirs',
                        default=['../data/ICDAR2019', '../data/ICDAR2017', '../data/ICDAR2015'],
                        help='Path to input directory')
    parser.add_argument('--ocr_input_list', nargs='?', type=str, default='../data/crop_words/gt.txt',
                        help='Path to crop word list for training recognition module')
    parser.add_argument('--input_size', nargs='?', type=int, default=512, help='Height of the input image')
    parser.add_argument('--max_iterators', nargs='?', type=int, default=400000, help='Max iterations')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4, help='Batch Size')
    parser.add_argument('--ocr_batch_size', nargs='?', type=int, default=512, help='Recognition Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--num_workers', nargs='?', type=int, default=5, help='# of workers')
    parser.add_argument('--debug', nargs='?', type=bool, default=False, help='Debug')
    parser.add_argument('--ocr_debug', nargs='?', type=bool, default=False, help='OCR Debug')
    parser.add_argument('--norm_height', nargs='?', type=int, default=44, help='Fixed Height for recognition module')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', nargs='?', type=str, default="checkpoints/LS1706203-320000.h5",
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()
    main(args)
