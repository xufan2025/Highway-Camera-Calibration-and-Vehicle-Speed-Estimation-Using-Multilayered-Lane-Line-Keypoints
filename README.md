# Highway Camera Calibration and Vehicle Speed Estimation Using Multilayered Lane-Line Keypoints
## Abstract：
Camera calibration enables the automatic estimation of intrinsic and extrinsic camera parameters, uncovering correspondences between 2D images and 3D real-world coordinates.For highway surveillance cameras, existing methods often rely on cumbersome procedures to extract limited priors (e.g., vanishing points or reference points) and provide incomplete estimations (e.g., roll angle). Therefore, we leverage the multilayered lanelines on highways, which offer rich priors such as segment lengths, intervals, and lane widths, to develop a novel camera
calibration and vehicle speed estimation method. For camera calibration, our approach performs road instance segmentation and extracts multilayered lane-line keypoints (MLK) while mitigating environmental interference and dynamic vehicle occlusions. An MLK-based calibration model is constructed and an angle polling Levenberg-Marquardt algorithm is designed to estimate key parameters, including focal length, three rotation angles, and lane-line distance. For vehicle speed estimation, multi-object tracking (MOT) algorithms are integrated with the calibration model to infer the average speeds of all identified vehicles. We collected real highway video footage from four different camera setups in Chinese highways. Experimental results demonstrate that our method outperforms existing methods across all setups. The impact of key parameters is evaluated to determine the optimal configuration. Lastly, its effectiveness in vehicle speed estimation is assessed using four advanced MOT algorithms.
## Installation

Conda virtual environment is recommended.

```bash
conda create -n venv1 python=3.9
conda activate venv1
pip install -r requirements1.txt
pip install -e .
```
```
  conda create -n venv2 python=3.7  
  conda activate venv2  
  pip install -r requirements2.txt  
  pip install -e .
```

## Main.py
  `python main.py`

## Datasets
  The dataset of Highway images can be downloaded from Baidu Webdisk.
  Link:https://pan.baidu.com/s/1kEaRAeRlK-ltE2beR3AvEQ   
  Extract code: kqzv

## Citation
Please cite this paper if you refer to our code or paper:  

>@article{xu2025highway,
  title={Highway Camera Calibration and Vehicle Speed Estimation Using Multilayered Lane-Line Keypoints},
  author={Xu, Fan and Zhai, Xiaoguang and Chen, Chuibin and Ma, Kai-Kuang and Wu, Qihui and Zhang, Xiaofei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}


