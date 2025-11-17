### Threshold Aware Cosine Similarity Driven Box Projection and Weighted Fusion for Confidence Recovery in Single Drone Tracking

This is the code showcasing the use of Circular Temporal WBF for Single Drone Tracking in various clear and noisy datasets.

## Setup
1. Model & Dataset  
    For demonstration of our paper we have used YOLOv5 here. The weights for YOLOv5 can be found in google drive link below:-  
    YOLOv5: https://drive.google.com/file/d/1k08PKmNOICoMIip3YkkkiZ_Lh1ygVmvP/view?usp=sharing  
    These weights were trained by https://github.com/AdnanMunir294/UAVD-CBRA  

    For the Dataset we have handpicked clips from videos from the following dataset:-  
    Drone-Tracking-Datasets: https://github.com/CenekAlbl/drone-tracking-datasets  

    We have then degraded these videos using degrade.py.  
    Both the handpicked clear and degraded videos can be found in the following google drive links:-  
    Clear Videos: https://drive.google.com/drive/folders/15PaPtinSy0UDxlwRJ6MaIZ9LMNtAeQ_S?usp=sharing  
    Degraded Videos: https://drive.google.com/drive/folders/13xlwrogW4a4zJesewEoiXkJjkJ-B5kOr?usp=sharing  

2. Environment & Dependencies  
    In a new directory run the following commands:

    ```bash
    python -m venv .venv
    .venv\Scripts\activate.bat
    git clone https://github.com/BhavyaK31/UAV-CIRCLE-WBF.git
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    cd ..
    ```

## Running
After you have downloaded the weights and dataset, and completed the setup, you need to set the following variables in main.py

```bash
input_folder = "path to dataset videos" 
output_folder = "path to where to store the resulting videos"
csv_output_path = "path to where to store the resulting csv file"
weights_path = "path to the YOLOv5 weigths"
```

you can then run main.py