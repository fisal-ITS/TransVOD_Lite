import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from PIL import Image
import requests
import matplotlib.pyplot as plt
import json
import jsonlines
import numpy as np

import torch
torch.set_grad_enabled(False)


 #LABELS = {1: 'airplane', 2: 'antelope', 3: 'bear', 4: 'bicycle', 5: 'bird',
#           6: 'bus', 7: 'car', 8: 'cattle', 9: 'dog', 10: 'domestic_cat',
#           11: 'elephant', 12: 'fox', 13: 'giant_panda', 14: 'hamster', 15: 'horse',
#           16: 'lion', 17: 'lizard', 18: 'monkey', 19: 'motorcycle', 20: 'rabbit',
#           21: 'red_panda', 22: 'sheep', 23: 'snake', 24: 'squirrel', 25: 'tiger',
#           26: 'train', 27: 'turtle', 28: 'watercraft', 29: 'whale', 30: 'zebra'}
LABELS = {1: 'crack'}                                                            

# colors for visualization
COLORS = [[231/256,141/256,141/256]]
#COLORS = [[231/256,141/256,141/256], [187/256,243/256,141/256], [187/256,106/256,221/256], [50/256,118/256,221/256],
#             [50/256,141/256,141/256], [50/256,243/256,141/256], [187/256,50/256,221/256], [50/256,118/256,50/256],
#             [50/256,50/256,150/256], [50/256,50/256,200/256], [50/256,100/256,150/256], [50/256,100/256,250/256],
#             [100/256,20/256,50/256], [150/256,20/256,20/256], [200/256,100/256,50/256], [250/256,150/256,150/256],
#             [50/256,100/256,0/256], [50/256,200/256,0/256], [100/256,250/256,0/256], [150/256,250/256,100/256],
#             [200/256,0/256,200/256], [150/256,0/256,250/256], [100/256,50/256,200/256], [50/256,0/256,250/256],
#             [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#             [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# COLORS = [[0.000, 0.447, 0.741], [2.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, scores, boxes, labels, out_path, savefig = False):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c, l in zip(scores, boxes, colors, labels):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[l], linewidth=13))
        text = f'{LABELS[l]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=40,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if savefig:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        return
    # plt.show()

def open_img(path):
    fp = open(path, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    pic = Image.fromarray(pic_array)
    pic_l = pic.convert("L")
    return pic_l

def visualize(path, out_path, scores, boxes, labels, prob_cutoff):
    pil_img = Image.open(path)
    #pil_img = open_img(path)
    plot_scores, plot_boxes, plot_labels = [], [], []
    for s, b, l in zip(scores, boxes, labels):
        if s > prob_cutoff:
            plot_scores.append(s)
            plot_boxes.append(b)
            plot_labels.append(l)
    plot_results(pil_img, plot_scores, plot_boxes, plot_labels, out_path, True)


def json_to_visual(file_path, out_dir, prob_cutoff, begin_number):
    with open(file_path, "r+", encoding="utf8") as f:
        count = 0
        for item in jsonlines.Reader(f):
            count += 1
            if count < begin_number:
                continue
            print(f'---processing pic#{count}---')
            path = item['path']
            #cur_path = 'data/vid/Data/VID/val'
            cur_path = 'content/drive/MyDrive/outputtrain1/val'        
            path_new = os.path.join(cur_path+'/'+path.split('/')[-2]+'/'+path.split('/')[-1]) 
            # import pdb
            # pdb.set_trace()                       
            out_path = os.path.join(out_dir, path.split('/')[-2]+'_'+path.split('/')[-1])
            path_a, path_b = os.path.split(out_path)
            # print('path', path)
            # print('path_new', path_new)
            # print('out_path', out_path) 
            # import pdb
            # pdb.set_trace()   

            visualize(path_new, out_path, item['scores'], item['boxes'], item['labels'], prob_cutoff)
            #count += 1


def main():
    if len(sys.argv) > 3:
        json_to_visual(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    else:
        json_to_visual('visual_result.json', 'output', 0.6)
    # path = 'dataset/dog.jpeg'
    # scores = [0.95, 0.32, 0.1]
    # boxes = [[428,35,619,201], [472,583,532,632], [13,35,315,314]]
    # labels = [3, 6, 1]
    # visualize(path, '', scores, boxes, labels, 0.6)


if __name__ == '__main__':
    main()
