import cv2
import numpy as np
import math
from collections import deque
from circle_wbf import circle_weighted_boxes_fusion

def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

def clip_box_to_frame(box, W, H):
    x1, y1, x2, y2 = box
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    x2 = np.clip(x2, 0, W - 1)
    y2 = np.clip(y2, 0, H - 1)
    return np.array([x1, y1, x2, y2], dtype=float)

def xyxy_to_norm(box, W, H):
    x1, y1, x2, y2 = box
    return [x1 / W, y1 / H, x2 / W, y2 / H]

def norm_to_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=float)

def run_directional_wbf_single_uav(
    model,
    video_path,
    weights_path,
    output_path="out_single_uav_wbf.mp4",
    conf_thresh=0.25,
    disp_thresh=20.0,        # movement threshold in pixels to trigger WBF
    base_shift_scale=0.3,    # baseline multiplier for generating candidates along displacement
    wbf_iou_thr=0.5,
    decay=1.1,
    window=5,
    yolo_thresh=0.9,
    cosine_thresh=0.25
):
    """
    Directional WBF for single UAV:
    - Keeps a window of previous fused boxes
    - Computes normalized historical displacements and their variance
    - Measures closeness (cosine similarity) between current displacement and history
    - Generates 3 directional candidates (nearer, predicted, farther) with perpendicular jitter
    - Adjusts candidate scales using past scale changes
    - Uses WBF to fuse candidates (weights based on YOLO conf and motion-consistency)
    - Falls back to YOLO output if displacement < disp_thresh
    """

    # history of fused boxes (xyxy arrays) and confidences
    prev_boxes = deque(maxlen=window)   # will store fused boxes (xyxy)
    prev_confs = deque(maxlen=window)   # corresponding confidences
    fused_ctb,yolo_ctb=[],[]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    wbf_count = 0
    frame_idx = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls?]

        if len(dets) == 0:
            out.write(frame)
            frame_idx += 1
            continue

        # single UAV -> choose highest confidence detection
        best = max(dets, key=lambda r: r[4])
        yolo_box = clip_box_to_frame(np.array(best[:4], float), W, H)
        yolo_conf = float(best[4])

        # if no history yet -> base case
        
        if len(prev_boxes) == 0:
            fused_box, fused_conf = yolo_box, yolo_conf
        else:
            # last fused center
            prev_center = box_center(prev_boxes[-1])
            curr_center = box_center(yolo_box)
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            disp = math.hypot(dx, dy)

            if disp < disp_thresh * math.hypot(yolo_box[0]-yolo_box[2],yolo_box[1]-yolo_box[3]) or yolo_conf>yolo_thresh:
                # small displacement -> trust YOLO directly
                fused_box, fused_conf = yolo_box, yolo_conf
                if (yolo_conf<=yolo_thresh):
                  prev_boxes.clear()
                  prev_confs.clear()
            else:
                # -------------------------
                # Compute historical displacement stats
                # -------------------------
                wbf_count += 1
                centers = [box_center(b) for b in prev_boxes]
                # compute consecutive displacements (from t-1 <- t-2 etc)
                hist_disps = []
                for i in range(1, len(centers)):
                    vx = centers[i][0] - centers[i-1][0]
                    vy = centers[i][1] - centers[i-1][1]
                    mag = math.hypot(vx, vy)
                    if mag > 1e-6:
                        hist_disps.append(np.array([vx / mag, vy / mag]))  # normalized direction
                # fallback if not enough history normalized vectors
                if len(hist_disps) == 0:
                    avg_norm_disp = np.array([dx / (disp + 1e-9), dy / (disp + 1e-9)])
                    disp_mags = [disp]
                else:
                    avg_norm_disp = np.mean(hist_disps, axis=0)
                    # re-normalize mean
                    norm = np.linalg.norm(avg_norm_disp) + 1e-9
                    avg_norm_disp = avg_norm_disp / norm
                    # historical magnitudes for jitter scaling
                    disp_mags = []
                    for i in range(1, len(centers)):
                        mag = math.hypot(centers[i][0] - centers[i-1][0], centers[i][1] - centers[i-1][1])
                        disp_mags.append(mag if mag > 1e-6 else 0.0)

                # historical normalized displacement for current displacement
                cur_norm = np.array([dx / (disp + 1e-9), dy / (disp + 1e-9)])

                # closeness: cosine similarity in [ -1, 1 ] -> map to [0,1]
                cos_sim = float(np.dot(avg_norm_disp, cur_norm))
                closeness = (cos_sim + 1.0) / 2.0
                closeness = max(0.0, min(1.0, closeness))

                # dispersion (std) of past magnitudes -> used to scale jitter
                if len(disp_mags) >= 2:
                    mag_std = float(np.std(disp_mags))
                    mag_mean = float(np.mean(disp_mags) + 1e-9)
                    mag_var_norm = mag_std / (mag_mean + 1e-9)
                else:
                    mag_var_norm = 0.0

                # size_scale derived from current YOLO box (characteristic length)
                yolo_w = float(yolo_box[2] - yolo_box[0])
                yolo_h = float(yolo_box[3] - yolo_box[1])
                size_scale = math.sqrt(max(yolo_w * yolo_h, 1.0))

                # perpendicular unit vector to current direction
                perp = np.array([-cur_norm[1], cur_norm[0]])
                # ensure unit
                perp_norm = perp / (np.linalg.norm(perp) + 1e-9)

                # -------------------------
                # decide beta (nearer / mid / farther) based on closeness
                # if closeness high -> closer candidates; if low -> farther candidates
                # -------------------------
                nearer_scale = 1.0 - 0.5 * closeness         # between 0.5 and 1.0
                mid_scale = 1.0
                farther_scale = 1.0 + 0.5 * (1.0 - closeness)  # between 1.0 and 1.5

                # jitter magnitude (pixels) based on history variance and size
                # If motion consistent (mag_var_norm small) -> small jitter. Else larger jitter.
                base_sigma = 0.12  # relative to size_scale; tuneable
                sigma = base_sigma * size_scale * (1.0 + mag_var_norm)
                # reduce jitter when closeness is high (we are confident in direction)
                sigma = sigma * (1.0 - 0.5 * closeness)

                # adjust scales for box size (zoom effects)
                # estimate simple scale ratio from previous areas if available
                area_ratios = []
                prev_areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in prev_boxes ]
                for i in range(1, len(prev_areas)):
                    denom = prev_areas[i-1] if prev_areas[i-1] > 0 else 1.0
                    area_ratios.append(prev_areas[i] / denom)
                if len(area_ratios) >= 1:
                    avg_area_ratio = float(np.mean(area_ratios))
                    # clamp
                    avg_area_ratio = max(0.85, min(1.15, avg_area_ratio))
                else:
                    avg_area_ratio = 1.0

                # candidate betas and scale multipliers
                betas = [nearer_scale, mid_scale, farther_scale]
                scale_mults = [0.98, 1.0 * avg_area_ratio, 1.02]  # small adjustments

                # generate candidate boxes: nearer, predicted, farther
                candidates_xyxy = []
                for b_idx, beta in enumerate(betas):
                    # predicted center from previous center plus scaled current displacement
                    cand_cx = prev_center[0] + beta * dx
                    cand_cy = prev_center[1] + beta * dy
                    # add perpendicular jitter: alternate sign for variety
                    sign = -1 if b_idx == 0 else (0 if b_idx == 1 else 1)
                    cand_cx += sign * perp_norm[0] * sigma
                    cand_cy += sign * perp_norm[1] * sigma

                    # candidate width/height
                    cand_w = max(2.0, yolo_w * scale_mults[b_idx])
                    cand_h = max(2.0, yolo_h * scale_mults[b_idx])

                    # convert center->xyxy
                    cand_x1 = cand_cx - cand_w / 2.0
                    cand_y1 = cand_cy - cand_h / 2.0
                    cand_x2 = cand_cx + cand_w / 2.0
                    cand_y2 = cand_cy + cand_h / 2.0

                    # clip to frame
                    cand_xyxy = clip_box_to_frame(np.array([cand_x1, cand_y1, cand_x2, cand_y2], dtype=float), W, H)
                    candidates_xyxy.append(cand_xyxy)

                # include the raw YOLO box as well (first)
                all_boxes_xyxy = [yolo_box] + candidates_xyxy

                # prepare normalized coords for WBF
                norm_boxes = [xyxy_to_norm(b, W, H) for b in all_boxes_xyxy]

                # compute weights: combine yolo_conf with motion consistency modifier
                # for each candidate compute its motion consistency = cosine with avg_norm_disp (mapped [0,1])
                motion_consistencies = []
                for b in all_boxes_xyxy:
                    c = box_center(b)
                    vx = c[0] - prev_center[0]
                    vy = c[1] - prev_center[1]
                    mag0 = math.hypot(vx, vy) + 1e-9
                    vnorm = np.array([vx / mag0, vy / mag0])
                    cos = float(np.dot(avg_norm_disp, vnorm))
                    #mc = (cos + 1.0) / 2.0 ###
                    mc = cos
                    mc = max(0.0, min(1.0, mc))
                    motion_consistencies.append(mc)

                # base weights: YOLO conf contributes; boost by motion_consistency for candidates (normalize later)
                base_w = yolo_conf
                weights = []
                for i_mc, mc in enumerate(motion_consistencies):
                    # Raw weight: base * (0.6 + 0.4*mc) – keeps baseline while preferring motion-consistent boxes
                    w = base_w * (np.sum(list(prev_confs))/len(prev_confs) + (1-np.sum(list(prev_confs))/len(prev_confs) * mc)) ####
                    # penalize farther candidate mildly (encourage nearer if closeness high)
                    weights.append(float(w))

                # ensure nonzero
                weights = [max(1e-4, w) for w in weights]

                labels = [0] * len(norm_boxes)  # single-class label 0 (keep consistent with WBF API)
                
                # run weighted boxes fusion; WBF expects list-of-lists per image (here one image)
                try:
                    fused_boxes, fused_scores, _ = circle_weighted_boxes_fusion(
                        [norm_boxes], [weights], [labels],
                        iou_thr=wbf_iou_thr, skip_box_thr=0.0001
                    )
                    if len(fused_boxes):
                        fused_box = norm_to_xyxy(fused_boxes[0], W, H)
                        fused_conf = float(fused_scores[0])
                    else:
                        fused_box, fused_conf = yolo_box, yolo_conf
                except Exception as e:
                    # fallback robustly to YOLO
                    fused_box, fused_conf = yolo_box, yolo_conf

        # Update history with fused result
        prev_boxes.append(fused_box)
        prev_confs.append(fused_conf)

        # Draw: YOLO (blue) and fused (red)
        yx1, yy1, yx2, yy2 = map(int, yolo_box)
        cv2.rectangle(frame, (yx1, yy1), (yx2, yy2), (255, 0, 0), 1)
        cv2.putText(frame, f"YOLO:{yolo_conf:.5f}", (yx1, yy2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        fx1, fy1, fx2, fy2 = map(int, fused_box)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        cv2.putText(frame, f"Fused:{fused_conf:.5f}", (fx1, fy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        out.write(frame)
        frame_idx += 1
        fused_ctb.append(fused_conf)
        yolo_ctb.append(yolo_conf)

    cap.release()
    out.release()
    print("Output saved:", output_path)
    print(f"Frames using wbf: {wbf_count}/{frame_idx}")
    return fused_ctb,yolo_ctb