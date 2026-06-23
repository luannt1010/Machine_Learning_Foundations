import torch
import cv2
import numpy as np
from PIL import Image
from src import batch_nms, decode_predictions
from src import YOLOv8
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image


def infer(model, image_path, conf_threshold=0.6, iou_threshold=0.7, imgsize=640,
          max_det=300, scale_to_original=False, return_cpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((imgsize, imgsize))

    img_np = np.array(resized)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        scores, _, boxes = decode_predictions(preds, imgsize)
        boxes, scores, labels = batch_nms(boxes, scores, confidence_threshold=conf_threshold, iou_threshold=iou_threshold, max_det=max_det)[0]

    if scale_to_original and boxes.numel() > 0:
        scale = torch.tensor([orig_w / imgsize, orig_h / imgsize, orig_w / imgsize, orig_h / imgsize], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes * scale

    if return_cpu:
        boxes = boxes.detach().cpu()
        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

    return boxes, scores, labels

def draw_bbox(image_path, boxes, labels):
    labels = [str(i.item()) for i in labels]
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1)
    result = vutils.draw_bounding_boxes(image=image,
                                        boxes=boxes,
                                        labels=labels,
                                        colors="red",
                                        width=4, font_size=30)
    result = to_pil_image(result)
    result.show()

if __name__ == "__main__":
    num_classes = 1
    state_dict_path = r"D:\private\yolov8\runs\train\last.pth"
    state_dict = torch.load(state_dict_path)
    model = YOLOv8(num_classes)
    model.load_state_dict(state_dict["model"])

    img_path = r"D:\private\anpr_system\test_img\tt.jpg"
    boxes, scores, labels = infer(model, img_path, scale_to_original=True, conf_threshold=0.1)
    draw_bbox(img_path, boxes, labels)