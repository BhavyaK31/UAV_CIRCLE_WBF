import math
import numpy as np

# ------- Geometry Conversions --------

def box_to_circle_xyxy(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    r = math.hypot(w / 2.0, h / 2.0)  # diagonal radius
    return cx, cy, r


# ------- Circle IoU (analytic) --------

def circle_intersection_area(x0, y0, r0, x1, y1, r1):
    d = math.hypot(x1 - x0, y1 - y0)
    if d >= r0 + r1:
        return 0.0
    if d <= abs(r0 - r1):
        return math.pi * min(r0, r1) ** 2

    r0s = r0 * r0
    r1s = r1 * r1
    alpha = math.acos((d * d + r0s - r1s) / (2 * d * r0))
    beta = math.acos((d * d + r1s - r0s) / (2 * d * r1))
    part = 0.5 * math.sqrt(
        (-d + r0 + r1) *
        (d + r0 - r1) *
        (d - r0 + r1) *
        (d + r0 + r1)
    )
    return r0s * alpha + r1s * beta - part


def circle_iou(c0, c1):
    x0, y0, r0 = c0
    x1, y1, r1 = c1
    inter = circle_intersection_area(x0, y0, r0, x1, y1, r1)
    union = math.pi * (r0 * r0 + r1 * r1) - inter
    if union <= 0:
        return 0.0
    return inter / union


# ------- Core Circular WBF --------

def circle_weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    mode="circle"   # ("circle" reserved; ellipse mode possible if needed)
):
    """
    Same API as weighted_boxes_fusion():
        boxes_list  -> list of N detector outputs of shape [num_boxes, 4] (norm xyxy)
        scores_list -> list of N score arrays [num_boxes]
        labels_list -> list of N label arrays [num_boxes]

    Returns:
        fused_boxes: Nx4 (normalized xyxy)
        fused_scores: N
        fused_labels: N
    """

    if weights is None:
        weights = [1.0] * len(boxes_list)

    # Flatten detections
    all_circles = []   # (cx, cy, r)
    all_scores = []
    all_labels = []
    all_weights = []

    for det_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        w_det = weights[det_idx]
        for box, score, lab in zip(boxes, scores, labels):
            if score < skip_box_thr:
                continue

            cx, cy, r = box_to_circle_xyxy(box)
            all_circles.append((cx, cy, r))
            all_scores.append(score)
            all_labels.append(lab)
            all_weights.append(w_det)

    all_circles = np.array(all_circles)
    all_scores = np.array(all_scores, dtype=float)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights, dtype=float)

    used = np.zeros(len(all_circles), dtype=bool)
    fused_boxes = []
    fused_scores = []
    fused_labels = []

    # Sort by scores descending
    order = np.argsort(-all_scores)

    for i in order:
        if used[i]:
            continue

        # Start cluster
        cluster = [i]
        used[i] = True

        for j in order:
            if used[j]:
                continue
            if all_labels[j] != all_labels[i]:
                continue
            if circle_iou(all_circles[i], all_circles[j]) >= iou_thr:
                cluster.append(j)
                used[j] = True

        cluster_scores = all_scores[cluster]
        cluster_weights = all_weights[cluster]
        total_weight = np.sum(cluster_scores * cluster_weights)
        if total_weight <= 0:
            weights_n = np.ones_like(cluster_scores) / len(cluster_scores)
        else:
            weights_n = (cluster_scores * cluster_weights) / total_weight

        # Fuse center & radius
        cxs = all_circles[cluster, 0]
        cys = all_circles[cluster, 1]
        rs  = all_circles[cluster, 2]

        fused_cx = float(np.sum(weights_n * cxs))
        fused_cy = float(np.sum(weights_n * cys))

        # Area-weighted for stability
        areas = np.pi * rs**2
        fused_area = float(np.sum(weights_n * areas))
        fused_r = math.sqrt(fused_area / math.pi)

        # Score = weighted
        fused_score = float(np.sum(weights_n * cluster_scores))
        fused_label = all_labels[i]

        # Convert circle → xyxy box
        x1 = fused_cx - fused_r
        y1 = fused_cy - fused_r
        x2 = fused_cx + fused_r
        y2 = fused_cy + fused_r

        fused_boxes.append([x1, y1, x2, y2])
        fused_scores.append(fused_score)
        fused_labels.append(fused_label)

    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)