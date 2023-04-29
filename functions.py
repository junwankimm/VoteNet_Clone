from util import *
import torch


#input : 3D bbox (6,) other set of 3D bbox(6,) since bbox is axis alingned only 6 coordinate is needed
#output : IOU, Scalar
def cal_iou2d(bb1, bb2):
    #bbox -> x1, y1, x2, y2 ~ (0,1)
    #겹치는 직사각형
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        iou = 0.
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    return iou

def cal_iou3d(bb1, bb2):
    #bbox : x1 y1 z1 x2 y2 z2 (000 -> 111)
    x_small = max(bb1[0], bb2[0])
    y_small = max(bb1[1], bb2[1])
    z_small = max(bb1[2], bb2[2])

    x_large = min(bb1[3], bb2[3])
    y_large = min(bb1[4], bb2[4])
    z_large = min(bb1[5], bb2[5])

    if x_large < x_small or y_large < y_small or z_large < z_small:
        iou = 0.
    else:
        intersection_volume = (x_large - x_small) * (y_large - y_small) * (z_large - z_small)
        bb1_volume = (bb1[3] - bb1[0]) * (bb1[4] - bb1[1]) * (bb1[5] - bb1[2])
        bb2_volume = (bb2[3] - bb2[0]) * (bb2[4] - bb2[1]) * (bb2[5] - bb2[2])

        iou = intersection_volume / (bb1_volume + bb2_volume - intersection_volume)

    return iou

#중심 박스에 대한 나머지 박스들의 IOU
def cal_iou3d_multi(box, boxes):
    x_small = boxes[:, 0].clamp(min=box[0])
    y_small = boxes[:, 1].clamp(min=box[1])
    z_small = boxes[:, 2].clamp(min=box[2])

    x_large = boxes[:, 3].clamp(max=box[3])
    y_large = boxes[:, 4].clamp(max=box[4])
    z_large = boxes[:, 5].clamp(max=box[5])

    x_delta = x_large - x_small
    y_delta = y_large - y_small
    z_delta = z_large - z_small

    iou = torch.zeros((len(boxes),), dtype=box.dtype)
    has_overlap = (x_delta > 0) * (y_delta > 0) * (z_delta > 0)

    if len(has_overlap.nonzero()) == 0:
        return iou
    else:
        boxes_valid = boxes[has_overlap]
        x_delta_valid = x_delta[has_overlap]
        y_delta_valid = y_delta[has_overlap]
        z_delta_valid = z_delta[has_overlap]

        intersection_volume = x_delta_valid * y_delta_valid * z_delta_valid
        box_volume = (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])
        boxes_volume = (boxes_valid[:, 3] - boxes_valid[:, 0]) * (boxes_valid[:, 4] - boxes_valid[:, 1]) * (boxes_valid[:, 5] - boxes_valid[:, 2])

        iou_valid = intersection_volume / (box_volume + boxes_volume -intersection_volume)
        iou[has_overlap] = iou_valid

    return iou짐

#input : set of bbox  Nx6, corresponding score N, iou threshold
#output : out boxes Mx6
def nms(boxes, scores, threshold):
    #score 높은순으로 정렬
    order = scores.argsort()
    keep = []
    while len(order) > 0:
        idx = order[-1]
        box = boxes[idx]
        keep.append(box)
        order = order[: -1]

        if len(order) == 0:
            break

        remaining_boxes = boxes[order]
        iou = cal_iou3d_multi(box, remaining_boxes)
        mask = iou < threshold
        order = order[mask]

    return torch.stack(keep)

if __name__ == '__main__':
    bb1 = torch.Tensor([0.1, 0.1, 0.3, 0.3])
    bb2 = torch.Tensor([0.2, 0.2, 0.4, 0.4])
    iou = cal_iou2d(bb1, bb2)
    bb1 = torch.Tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    bb2 = torch.Tensor([0.2, 0.2, 0.2, 0.4, 0.4, 0.4])
    iou = cal_iou3d(bb1, bb2)

    box = torch.tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    boxes = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4], [0.01, 0.01, 0.01, 0.03, 0.03, 0.03]])
    iou = cal_iou3d_multi(box, boxes)

    boxes = torch.tensor([
        [0.1,0.1,0.1,0.3,0.3,0.3],
        [0.11, 0.11, 0.11, 0.31, 0.31, 0.31],
        [0.2, 0.2, 0.2, 0.4, 0.4, 0.4],
        [0.21, 0.21, 0.21, 0.4, 0.4, 0.4],
    ])
    scores = torch.tensor([0.9, 0.6, 0.7, 0.6])
    nms_boxes = nms(boxes, scores, threshold=0.5)
    print(nms_boxes)


