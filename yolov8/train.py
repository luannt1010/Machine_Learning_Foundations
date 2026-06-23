import argparse
import torch
from src import yolo_collate_fn, YOLOv8Dataset
from src import get_model
from src import YOLOv8Loss
from src import train
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=r"D:\private\yolov8\Vietnam license-plate.v1i.yolov8\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\private\yolov8\Vietnam license-plate.v1i.yolov8\valid")
    parser.add_argument("--sp", type=str, default="runs/train")

    parser.add_argument("--suffix", type=str, default="n")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--reg_max", type=int, default=16)

    return parser.parse_args()

def main():
    args = get_args()

    batch_size = args.batch_size
    tr_root_dir = args.train_dir
    val_root_dir = args.val_dir
    tr_dataset = YOLOv8Dataset(tr_root_dir)
    val_dataset = YOLOv8Dataset(val_root_dir)
    tr_loader = DataLoader(tr_dataset, shuffle=True, collate_fn=yolo_collate_fn, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=yolo_collate_fn, batch_size=batch_size)
    print(f"Length of train dataset: {len(tr_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")

    num_classes = max(tr_dataset.num_classes, val_dataset.num_classes)
    epochs = args.epochs
    sp = args.sp
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    suffix = args.suffix
    model = get_model(num_classes, suffix, args.reg_max)

    print(f"Total arguments: {sum([p.numel() for p in model.parameters()])}")

    loss_fn = YOLOv8Loss(num_classes, imgsize=tr_dataset.imgsize)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    train(model, tr_loader, val_loader, loss_fn, optimizer, epochs, sp)

if __name__ == "__main__":
    main()
