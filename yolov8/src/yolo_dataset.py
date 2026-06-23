import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def yolo_collate_fn(batch):
    images = []
    labels_list = []
    boxes_list = []
    for sample in batch:
        image, labels, boxes = sample
        images.append(image)
        labels_list.append(labels)
        boxes_list.append(boxes)
    images = torch.stack(images, 0)
    B = len(images)
    max_gt = 0
    for label in labels_list:
        max_gt = max(max_gt, len(label))
    if max_gt == 0:
        max_gt = 1
    gt_labels = torch.zeros((B, max_gt), dtype=torch.long)
    gt_bboxes = torch.zeros((B, max_gt, 4), dtype=torch.float32)
    mask_gt = torch.zeros((B, max_gt), dtype=torch.bool)
    for i in range(B):
        n = len(labels_list[i])
        if n == 0:
            continue
        gt_labels[i, :n] = labels_list[i].long()
        gt_bboxes[i, :n] = boxes_list[i].float()
        mask_gt[i, :n] = True
    return images, gt_labels, gt_bboxes, mask_gt

class YOLOv8Dataset(Dataset):
    def __init__(self, root_dir, imgsize=640, transform=None):

        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "labels")
        self.transform = transform
        self.imgsize = self.calc_imgsize(imgsize)
        self.images = self.find_all_images()
        self.num_classes = self.find_all_classes()

    def __len__(self):
        return len(self.images)

    def find_all_classes(self):
        res = set()
        for ann_name in os.listdir(self.ann_dir):
            ann_path = os.path.join(self.ann_dir, ann_name)
            with open(ann_path, "r") as f:
                for line in f:
                    line = line.split()
                    label = int(line[0])
                    if label not in res:
                        res.add(label)
        return len(res)

    def parses_annotation(self, ann_file, img_w, img_h):
        labels = []
        bboxes = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split()
                label = int(line[0])
                xywh = [float(i) for i in line[1:]]
                cx, cy, w, h = xywh
                cx *= img_w
                cy *= img_h
                w *= img_w
                h *= img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                bbox = [x1, y1, x2, y2]
                labels.append(label)
                bboxes.append(bbox)
        return torch.tensor(labels, dtype=torch.int64), torch.tensor(bboxes, dtype=torch.float32)

    def calc_imgsize(self, imgsize):
        if imgsize % 32 != 0:
            factor = 32 - (imgsize % 32)
            imgsize += factor
        return imgsize

    def find_all_images(self):
        img_names = sorted(os.listdir(self.img_dir))
        images = []
        for img_name in img_names:
            if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_path = os.path.join(self.img_dir, img_name)
                images.append(img_path)
        return images

    def __getitem__(self, idx):
        image_path = self.images[idx]
        name_img = os.path.basename(image_path)
        name_ann = os.path.splitext(name_img)[0] + ".txt"
        ann_file = os.path.join(self.ann_dir, name_ann)
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        labels, bboxes = self.parses_annotation(ann_file, img_w, img_h)
        new_size = self.imgsize
        image = image.resize((new_size, new_size))
        if len(bboxes) != 0:
            scale_x = new_size / img_w
            scale_y = new_size / img_h
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y

        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, labels, bboxes


# dataset = YOLOv8Dataset(
#     root_dir=r"D:\private\yolov8\Vietnam license-plate.v1i.yolov8\train",
#     imgsize=640
# )
#
# image, labels, boxes = dataset[0]
#
# print(dataset.num_classes)
# print(image.shape)
# print(labels.shape, labels)
# print(boxes.shape, boxes)


