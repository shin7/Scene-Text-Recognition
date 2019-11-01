import os
import cv2
import time
import random
import numpy as np
import unicodedata as ud
from datasets import data_util
import torchvision.transforms as transforms
import PIL

buckets = []
for i in range(1, 100):
    buckets.append(8 + 4 * i)

f = open('files/codec.txt', 'r')
codec = f.readlines()[0]
codec_rev = {}
index = 4
for i in range(0, len(codec)):
    codec_rev[codec[i]] = index
    index += 1
f.close()


def read_image_list(data_path):
    base_dir = os.path.dirname(data_path)
    files_out = []
    cnt = 0
    with open(data_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\ufeff', '')
            line = line.strip()
            if len(line) == 0:
                continue
            if not line[0] == '/':
                line = '{0}/{1}'.format(base_dir, line)
            files_out.append(line)
            cnt += 1
    return files_out


def generator(input_list, batch_size=4, norm_height=32):
    image_list = np.array(read_image_list(input_list))
    print('{} training images in {}'.format(image_list.shape[0], input_list))
    index = np.arange(0, image_list.shape[0])

    transform = transforms.Compose([
        transforms.ColorJitter(.3, .3, .3, .3),
        transforms.RandomGrayscale(p=0.1)
    ])

    batch_sizes = []
    cb = batch_size
    for i in range(0, len(buckets)):
        batch_sizes.append(cb)
        if i % 10 == 0 and cb > 2:
            cb /= 2

    max_samples = len(image_list) - 1
    bucket_images = []
    bucket_labels = []
    bucket_label_len = []

    for b in range(0, len(buckets)):
        bucket_images.append([])
        bucket_labels.append([])
        bucket_label_len.append([])

    while True:
        np.random.shuffle(index)

        for i in index:
            try:
                image_name = image_list[i]

                src_del = " "
                spl = image_name.split(" ")
                if len(spl) == 1:
                    spl = image_name.split(",")
                    src_del = ","
                image_name = spl[0].strip()
                gt_txt = ''
                if len(spl) > 1:
                    gt_txt = ""
                    delim = ""
                    for k in range(2, len(spl)):
                        gt_txt += delim + spl[k]
                        delim = src_del
                    if len(gt_txt) > 1 and gt_txt[0] == '"' and gt_txt[-1] == '"':
                        gt_txt = gt_txt[1:-1]

                # print(gt_txt)

                if len(gt_txt) == 0:
                    continue

                if image_name[len(image_name) - 1] == ',':
                    image_name = image_name[0:-1]

                if not os.path.exists(image_name):
                    continue

                im = cv2.imread(image_name)
                if im is None:
                    continue

                if im.shape[0] > im.shape[1] and len(gt_txt) > 4:
                    # cv2.imshow('bad', im)
                    # print(image_name)
                    # cv2.waitKey(0)
                    continue

                scale = norm_height / float(im.shape[0])
                width = int(im.shape[1] * scale) + random.randint(- 2 * norm_height, 2 * norm_height)

                best_diff = width
                bestb = 0
                for b in range(0, len(buckets)):
                    if best_diff > abs(width - buckets[b]):
                        best_diff = abs(width - buckets[b])
                        bestb = b

                if random.randint(0, 100) < 10:
                    bestb += random.randint(-1, 1)
                    bestb = max(0, bestb)
                    bestb = min(bestb, (len(buckets) - 1))

                width = buckets[bestb]
                im = cv2.resize(im, (int(buckets[bestb]), norm_height))

                if random.randint(0, 100) < 10:
                    im = np.invert(im)
                if random.randint(0, 100) < 10:
                    im = cv2.blur(im, (3, 3))

                if random.randint(0, 100) < 10:
                    warp_mat = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), 0, 1)
                    warp_mat[0, 1] = random.uniform(-0.1, 0.1)
                    im = cv2.warpAffine(im, warp_mat, (im.shape[1], im.shape[0]))

                pim = PIL.Image.fromarray(np.uint8(im))
                pim = transform(pim)

                im = np.array(pim)

                bucket_images[bestb].append(im[:, :, :].astype(np.float32))

                gt_labels = []
                for k in range(len(gt_txt)):
                    if gt_txt[k] in codec_rev:
                        gt_labels.append(codec_rev[gt_txt[k]])
                    else:
                        # print('Unknown char: {0}'.format(gt_txt[k]))
                        gt_labels.append(3)

                if 'ARABIC' in ud.name(gt_txt[0]):
                    gt_labels = gt_labels[::-1]

                bucket_labels[bestb].extend(gt_labels)
                bucket_label_len[bestb].append(len(gt_labels))

                if len(bucket_images[bestb]) == batch_sizes[bestb]:
                    images = np.asarray(bucket_images[bestb], dtype=np.float)
                    images /= 128
                    images -= 1

                    yield images, bucket_labels[bestb], bucket_label_len[bestb]
                    max_samples += 1
                    max_samples = min(max_samples, len(image_list) - 1)
                    bucket_images[bestb] = []
                    bucket_labels[bestb] = []
                    bucket_label_len[bestb] = []

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = data_util.GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':

    data_generator = get_batch(input_list="../data/crop_words/gt.txt", num_workers=1, batch_size=1)
    while True:
        data = next(data_generator)
        print(data)
