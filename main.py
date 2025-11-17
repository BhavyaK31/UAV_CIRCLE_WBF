import os
import csv
import torch
import numpy as np
from temporal_wbf import run_directional_wbf_single_uav

input_folder = "./dataset/blurred" 
output_folder = "./result/blurred/videos"
csv_output_path = "./result/blurred/results.csv"

os.makedirs(output_folder, exist_ok=True)

# load YOLO model
weights_path = "./Yolov5_best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
model.to(device)

csv_fields = [
    "video_name",
    "frames_fused_better",
    "frames_fused_equal",
    "avg_accuracy_increase_wbf_only",
    "avg_accuracy_increase_over_video",
    "avg_yolo_conf",
    "avg_fused_conf",
    "avg_yolo_conf_all",
    "avg_fused_conf_all"
]


csv_file = open(csv_output_path, "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
writer.writeheader()

for file in os.listdir(input_folder):
    if file.lower().endswith(".mp4"):
        video_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, f"OUT_{file}")

        fused_ctb, yolo_ctb, wbf_used, num_frames = run_directional_wbf_single_uav(model, 
                                                                                   video_path=video_path, 
                                                                                   weights_path=weights_path, 
                                                                                   output_path=out_path, 
                                                                                   disp_thresh=0.1, 
                                                                                   window=10, 
                                                                                   conf_thresh=1)

        # ==== METRICS ====
        c_better = 0
        c_equal = 0

        wbf_improve_vals = []
        full_improve_vals = []

        for f, y in zip(fused_ctb, yolo_ctb):
            diff = f - y
            if diff > 0:
                c_better += 1
                full_improve_vals.append(diff)
            elif diff == 0:
                c_equal += 1
            else:
                full_improve_vals.append(diff)

            # only when fused was actually different
            if diff != 0:
                wbf_improve_vals.append(diff)

        avg_wbf_increase = np.sum(wbf_improve_vals)/c_better if c_better != 0 else 0
        avg_full = np.sum(wbf_improve_vals)/(c_better + c_equal) if (c_better + c_equal) != 0 else 0
        
        writer.writerow({
            "video_name": file,
            "frames_fused_better": c_better,
            "frames_fused_equal": c_equal,
            "avg_accuracy_increase_wbf_only": avg_wbf_increase,
            "avg_accuracy_increase_over_video": avg_full,
            "avg_yolo_conf": np.mean(yolo_ctb),
            "avg_fused_conf": np.mean(fused_ctb),
            "avg_yolo_conf_all": np.sum(yolo_ctb) / num_frames,
            "avg_fused_conf_all": np.sum(fused_ctb) / num_frames
        })

csv_file.close()
print("DONE: All videos processed and CSV saved to:", csv_output_path)