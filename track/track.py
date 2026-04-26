import argparse
import cmath
import os
import operator as operator
from PIL import Image
from numpy import vstack
from sympy.physics.continuum_mechanics.beam import numpy
# limit the number of cpus used by high performance libraries
from sympy import symbols, Eq, solve

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import csv
from collections import defaultdict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  #
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.augmentations import letterbox
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # 读取视频帧率和分辨率
    video_path = source
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wide = size[0]
    height = size[1]
    # t_fps = 1 / fps
    t_fps = 1 / fps
    t_25 = t_fps * 50

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources
    outputs_prev = []
    result_51 = []
    data = []

    count = np.zeros((len(names)))  # 构建一个与类别数相同的数组，用于统计各个类别数量
    id_latest = 0
    file_handle = open("D:\chen_pythonfile\Yolov5_StrongSORT_OSNet-master/1.txt", mode='w')  # 打开一个记录id的txt

    # Run tracking
    with open(data_path) as f:
        for line in f:
            line = line.strip('\n')
            line = float(line)
            data.append(line)
    f = data[0]
    theta = data[1]
    k1 = data[2]
    k2 = data[3]
    b1 = data[4]
    b2 = data[5]
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    id_speeds = defaultdict(list)  # id -> list of speeds for each frame
    seen_ids = set()

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # Crop to lower 3/4 (bottom 3/4, ignore top 1/4)
        im0_list = im0s if isinstance(im0s, list) else [im0s]
        cropped_im0_list = []
        new_im_list = []
        for orig_im0 in im0_list:
            h, w, c = orig_im0.shape
            crop_y = h // 4
            cropped = orig_im0[crop_y:, :, :]
            cropped_im0_list.append(cropped)
            lb_img = letterbox(cropped, imgsz, stride=stride, auto=pt)[0]
            lb_img = lb_img.transpose((2, 0, 1))[::-1]
            lb_img = np.ascontiguousarray(lb_img)
            new_im_list.append(lb_img)
        im = np.stack(new_im_list, 0) if len(new_im_list) > 1 else new_im_list[0]

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                   max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        updated_ids = set()
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to cropped_im0 size
                cropped_shape = cropped_im0_list[i].shape
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], cropped_shape).round()
                # Adjust y coordinates back to full frame
                crop_y = im0_list[i].shape[0] // 4
                det[:, 1] += crop_y
                det[:, 3] += crop_y

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                if len(outputs_prev) > 0:
                    outputs_prev = np.array(outputs_prev).reshape(np.array(outputs_prev).shape[1], 7)
                t5 = time_sync()
                dt[3] += t5 - t4

                id_latest_prev = id_latest  # 标记前一帧的最新id，若当前帧存在新id则加入到统计中
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        file_handle.writelines(
                            ['id:', str(int(id)), ' ', names[int(cls)], ' ', str(bboxes), ' '])  # 记录id

                        if id > id_latest_prev:
                            count[int(cls)] = count[int(cls)] + 1
                            if id > id_latest:
                                id_latest = id
                        V = 0
                        X = 0
                        if len(result_51) == 51:
                            result1 = result_51[0]
                            result1 = np.array(result1)
                            result1 = np.array(result1).reshape(np.array(result1).shape[1], 7)
                            for q in range(result1.shape[0]):
                                if id == result1[q, 4]:
                                    try:
                                        V = \
                                        cesu((output[2] + output[0]) / 2, output[3], result1[q, 3], H, D, n_lane, t_25,
                                             f, theta, wide, height, k1, k2, b1, b2)[0]
                                    except:
                                        pass

                        # Update speeds
                        if id not in seen_ids:
                            seen_ids.add(id)
                            # Fill previous frames with 0
                            id_speeds[id].extend([0.0] * frame_idx)

                        id_speeds[id].append(round(V, 2))
                        updated_ids.add(id)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]} {V}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f} {V}' if hide_class else f'{id} {names[c]} {conf:.2f} {V:.2f}'))
                            # annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')
                outputs = [[]]
            file_handle.writelines('\n')  # 记录id

            # Append 0 for ids not updated in this frame
            for sid in seen_ids - updated_ids:
                id_speeds[sid].append(0.0)

            # Stream results
            im0 = annotator.result()

            # 打印统计信息
            x_list = 50
            y_list = 200
            for i_cls in range(len(names)):
                line_list = str(names[int(i_cls)]) + ': ' + str(int(count[int(i_cls)]))
                cv2.putText(im0, line_list, (x_list, y_list), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                y_list = y_list + 50

            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
                prev_frames[i] = curr_frames[i]
            prev1 = im0.copy()
        if len(result_51) < 51:
            result_51.append(outputs.copy())
        else:
            del result_51[0]
            result_51.append(outputs.copy())
        outputs_prev = outputs.copy()

    file_handle.close()  # 测试：输出id
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    # Save to CSV files
    output_dir = "C:/Users/zgshang/Desktop/50/"
    for id_val, speeds in id_speeds.items():
        csv_path = f"{output_dir}id_{id_val}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'speed'])
            for frame, speed in enumerate(speeds):
                writer.writerow([frame, speed])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def cesu(x1, y1, y2, H, D, n_lane, t_fps, f, theta, wide, height, k1, k2, b1, b2):
    a1 = k1 * y1 + b1
    a2 = k2 * y1 + b2
    c1 = k1 * y2 + b1
    c2 = k2 * y2 + b2
    l1 = abs(c1 - c2)
    l2 = abs(a1 - a2)
    y_prev = y2 - height / 2
    y_after = y1 - height / 2

    A1 = np.array([1, 0])
    A2 = np.array([1, theta])
    A = vstack((A1, A2))
    d1 = np.array([cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y_prev ** 2)) / (l1 ** 2 * theta ** 2)) - H ** 2)])
    d2 = np.array([cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y_after ** 2)) / (l2 ** 2 * theta ** 2)) - H ** 2)])
    d = vstack((d1, d2))
    A_1 = np.linalg.pinv(A)
    r = np.dot(A_1, d)
    Y = abs(r[0, 0])
    C = abs(r[1, 0])
    v = (C * 3.6) / (1000 * t_fps)
    X1 = (x1 - wide / 2) * (cmath.sqrt(H ** 2 + Y ** 2) / cmath.sqrt(f ** 2 + (y_after / 2) ** 2))
    X1 = X1.real / 1000
    return v, X1


if __name__ == "__main__":
    data_path = r"MOT16_eval/1.txt"  #path to the fitted parameters
    H = 6000
    D = 3750
    C = 15000
    n_lane = 2
    opt = parse_opt()
    main(opt)