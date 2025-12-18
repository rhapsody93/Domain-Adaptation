import os, glob, json, yaml, shutil
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression  # NMS for raw model outputs


# ===========================================================
# CONFIGURATION
# ===========================================================
PROJECT = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/domain_adaptation/self_training'

# Source Domain
SD_YAML = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/source_domain/train/merged_dataset.yaml'

# Target Domain
TD_YAML = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/target_domain/train/merged_dataset.yaml'
TD_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/target_domain_weather/MERGED_TD_BB'
TD_IMAGES_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/target_domain_weather/MERGED_TD_BB/images/train'
TD_TEST_IMAGES_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/target_domain_weather/MERGED_TD_BB/images/test'

# AI-Hub JSON Metadata
AIHUB_JSON_BASE = '/home/jhkim/anaconda3/envs/project_ms/original_datasets/target_domain_weather/adverse_weather_datasets/AI-Hub/autonomous_driving_adverse_weather/open_source_data/Training/02.labeling_data/TL/2D'

# Model & Output
SD_WEIGHT_DIR = f"{PROJECT}/sd_train/weights"
os.makedirs(SD_WEIGHT_DIR, exist_ok=True)

TD_PSEUDO_LABEL_DIR      = f"{PROJECT}/pseudo_labels/td_train_txt"
TD_PSEUDO_VIS_DIR        = f"{PROJECT}/pseudo_labels/td_train_vis"
TD_PSEUDO_LABEL_TEST_DIR = f"{PROJECT}/pseudo_labels/td_test_txt"
TD_PSEUDO_VIS_TEST_DIR   = f"{PROJECT}/pseudo_labels/td_test_vis"

# Shadow merged dataset (이미지/라벨 심볼릭 링크를 모아둘 안전한 병합 데이터셋)
MERGED_DATASET_DIR = f"{PROJECT}/merged_sd_td_pseudo"
MERGED_YAML = f"{PROJECT}/merged_with_pseudo.yaml"

for d in [TD_PSEUDO_LABEL_DIR, TD_PSEUDO_VIS_DIR, TD_PSEUDO_LABEL_TEST_DIR, TD_PSEUDO_VIS_TEST_DIR, MERGED_DATASET_DIR]:
    os.makedirs(d, exist_ok=True)

# Training params
IMG_SIZE = 640
BATCH = 8
EPOCHS = 100

# ---------- WP-PL-UC hyperparams ----------
MC_ITERS = 20
TEMP = 1.2
IOU_MATCH_THR = 0.5

ENTROPY_THR = 1.2
UMC_STD_THR = 0.04
BVAR_THR    = 0.01

BASE_CONF_THR = 0.15
UW_SAMPLE_MAX = 0   # >0 로 설정하면 Uw 샘플링 수를 제한

# ===========================================================
# CLASS REMAPPING
# ===========================================================
SD_CLASSES = ["person", "bicycle", "car", "motorcycle", "traffic light", "bus", "train", "truck", "traffic sign"]
TD_CLASSES = ["person", "bicycle", "car", "motorcycle", "traffic light", "bus", "train", "truck", "traffic sign"]

# ID identical, safe explicit mapping
SD_TO_TD_IDX = {
    0:0, 1:1, 2:2, 3:3, 4:4,
    5:5, 6:6, 7:7, 8:8
}
SD_TO_TD_IDX = {i: i for i, cls in enumerate(SD_CLASSES) if cls in TD_CLASSES}

# 날씨 키워드
WEATHER_KEYWORDS = ["fog", "rain", "snow", "sand", "night", "cloudy"]


def check_yaml_structure(yaml_path, name="YAML"):
    print(f"\n### [CHECK] Checking {name}: {yaml_path}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"[ERROR] {name} does not exist: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "train" not in cfg:
        raise ValueError(f"[ERROR] {name} missing 'train' key")

    print(f"[CHECK] {name} OK. keys={list(cfg.keys())}")


def check_class_mapping():
    print("\n===== [CHECK] CLASS MAPPING =====")
    print("SD_CLASSES:", SD_CLASSES)
    print("TD_CLASSES:", TD_CLASSES)
    print("SD_TO_TD_IDX:", SD_TO_TD_IDX)
    print("=================================\n")


def check_weather_labels_sample(img_dir, n=10):
    print("\n===== [CHECK] Weather sample =====")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
    for img in imgs:
        print(f"{img} -> {get_weather_label(img)}")
    print("=================================\n")


def check_gpu_memory():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024**3)
        print(f"[CHECK] GPU Memory = {total:.2f} GB")
        if total < 12:
            print("[WARN] GPU <12GB: YOLOv8-L + MC Dropout 위험")
    else:
        print("[WARN] CUDA not available")


def check_dropout_duplicate(model):
    count = 0
    for _, m in model.model.model.named_modules():
        if hasattr(m, "_dropout_patched"):
            count += 1
    print(f"[CHECK] Dropout patched count = {count}")
    if count > 12:
        print("[WARN] Dropout 중복 패치 가능성 있음")


def log_pseudo_stats(txt_dir):
    txts = glob.glob(os.path.join(txt_dir, "*.txt"))
    counts = []
    for t in txts:
        with open(t, "r") as f:
            lines = f.readlines()
            counts.append(len(lines))

    counts = np.array(counts)
    print(f"[PL-LOG] Total pseudo files: {len(txts)}")
    print(f"[PL-LOG] Non-empty: {(counts > 0).sum()} ({(counts > 0).sum() / len(counts):.2%})")
    print(f"[PL-LOG] Mean boxes: {counts.mean():.2f} | Std: {counts.std():.2f}")
    print(f"[PL-LOG] Max: {counts.max()}, Min: {counts.min()}")


def _has_dropout(m: nn.Module) -> bool:
    # parents 모듈 아래 'Dropout' 계층이 하나라도 있으면 True
    return any(isinstance(sub, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
               for _, sub in m.named_modules() if _ != "")


# ===========================================================
# DROPOUT INJECTION (forward 패치 버전)
# ===========================================================
def _patch_with_dropout(module: nn.Module, p: float):
    """
    모듈 타입은 건드리지 않고, forward 끝에 2D dropout을 추가한다.
    - DetectionModel 의 구조(f, i 등)는 유지
    - Conv/BN 등 내부 서브모듈도 그대로 유지
    """
    if hasattr(module, "_dropout_patched"):
        # 이미 패치된 경우 중복 방지
        return

    old_forward = module.forward

    def new_forward(x, *args, old_forward=old_forward, p=p, **kwargs):
        out = old_forward(x, *args, **kwargs)

        # 텐서가 아닐 경우는 그대로 반환 (ex. tuple 등)
        if not torch.is_tensor(out):
            return out

        if out.dim() == 4:
            # N, C, H, W → spatial dropout
            return F.dropout2d(out, p=p, training=True)
        elif out.dim() == 3:
            # N, C, L → channel-wise 1D dropout
            return F.dropout1d(out, p=p, training=True)
        else:
            # 기타 (2D, 1D 등) → 일반 dropout
            return F.dropout2d(out, p=p, training=True)

    module.forward = new_forward
    module._dropout_patched = True
    module._dropout_p = p


def _patch_detect_head_logits(detect, p_head):
    """
    YOLOv8 Detect(head) 의 forward 를 patch하여
    raw detection logits 에 dropout 주입.

    detect.forward() 출력을 완전 안전하게 처리:
      - Tensor 반환
      - (pred,) 형태
      - (pred, aux) 형태
      - (pred, anchors, stride) 등 다양한 YOLO head 변형 대비
    """
    if hasattr(detect, "_dropout_patched_head"):
        return

    old_forward = detect.forward

    def new_forward(x, *args, **kwargs):
        out = old_forward(x, *args, **kwargs)

        # ===== 케이스 1: Tensor 단독 =====
        if isinstance(out, torch.Tensor):
            return F.dropout(out, p=p_head, training=True)

        # ===== 케이스 2: tuple/list 반환 =====
        if isinstance(out, (tuple, list)):
            new_out = []
            for item in out:
                if isinstance(item, torch.Tensor):
                    new_out.append(F.dropout(item, p=p_head, training=True))
                else:
                    new_out.append(item)
            return tuple(new_out)

        # ===== 케이스 3: 예상 밖 타입 =====
        raise TypeError(f"[ERROR] detect.forward returned unsupported type: {type(out)}")

    detect.forward = new_forward
    detect._dropout_patched_head = True
    detect._dropout_p_head = p_head


def add_dropout_to_yolov8(model, p_backbone=0.1, p_neck=0.1, p_head=0.3) -> YOLO:
    """
    Conv2d 나 C2f 모듈 타입을 바꾸지 않고,
    backbone/neck 블록의 forward 끝에 dropout을 삽입하는 방식.

    backbone/neck forward 끝에 dropout 패치 + head(cls/obj) branch dropout 추가.

    - backbone: model[6], [8], [9]   (C2f, C2f, SPPF)
    - neck:     model[12], [15], [18], [21]  (C2f)
    - head(model[22])는 full detect head 적용(detect.cv2 + detect.cv3 + detect.dfl + logits 작동)
    """
    seq = model.model.model  # DetectionModel.model: Sequential

    backbone_ids = [6, 8, 9]
    neck_ids     = [12, 15, 18, 21]
    head_id = 22

    # backbone dropout patch
    for idx in backbone_ids:
        if 0 <= idx < len(seq):
            _patch_with_dropout(seq[idx], p_backbone)

    # neck dropout patch
    for idx in neck_ids:
        if 0 <= idx < len(seq):
            _patch_with_dropout(seq[idx], p_neck)

    # head dropout patch (cv2, cv3, dfl + logits)
    #if head_id < len(seq):
    detect = seq[head_id]

    # Dropout on cv2 (classification/objectness convs)
    if hasattr(detect, "cv2"):
        _patch_with_dropout(detect.cv2, p_head)

    if hasattr(detect, "cv3"):
        _patch_with_dropout(detect.cv3, p_head)

    if hasattr(detect, "dfl"):  # DFL branch (important for box refinement)
        _patch_with_dropout(detect.dfl, p_head)

    _patch_detect_head_logits(detect, p_head)

    print("[INFO] Dropout correctly inserted ONLY in YOLOv8 backbone+neck+head.")
    return model


# ===========================================================
# UTILS
# ===========================================================
def apply_temperature_scaling(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)


def binary_entropy_from_conf(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def xywh_to_xyxy(boxes_xywh: np.ndarray):
    """
    boxes_xywh: (N,4) [cx, cy, w, h]
    return: (N,4) [x1, y1, x2, y2]
    """
    if boxes_xywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x_c, y_c, w, h = (
        boxes_xywh[:, 0],
        boxes_xywh[:, 1],
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    )
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

# ===========================================================
# WEATHER MAPPING (fog, rain, snow, sand, night, cloudy)
# ===========================================================
def _normalize_weather_key(w):
    if w is None:
        return "unknown"
    s = str(w).strip().lower()
    if any(k in s for k in ["fog", "foggy", "mist", "haze", "안개"]):
        return "fog"
    if any(k in s for k in ["rain", "비"]):
        return "rain"
    if any(k in s for k in ["snow", "눈"]):
        return "snow"
    if any(k in s for k in ["sand", "dust", "모래"]):
        return "sand"
    if any(k in s for k in ["night", "야간"]):
        return "night"
    if any(k in s for k in ["흐림", "cloud", "cloudy", "overcast"]):
        return "cloudy"
    if "맑" in s or "clear" in s:
        return "clear"
    return "unknown"


def extract_weather_from_aih_json(img_name):
    parts = img_name.split("_")
    if len(parts) < 3:
        return "unknown"
    scene_id = "_".join(parts[:3])
    search_path = os.path.join(AIHUB_JSON_BASE, scene_id, "sensor_raw_data/camera")
    if not os.path.exists(search_path):
        return "unknown"
    for jf in glob.glob(os.path.join(search_path, "*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                weather_str = data.get("Environment_meta", {}).get("weather", "")
                return _normalize_weather_key(weather_str)
        except:
            continue
    return "unknown"


def get_weather_label(img_path):
    fname = os.path.basename(img_path).lower()
    if any(k in fname for k in ["fog_", "foggy", "mist", "haze"]):
        return "fog"
    if any(k in fname for k in ["rain_", "rain", "rain_storm"]):
        return "rain"
    if any(k in fname for k in ["snow_", "snow", "snow_storm"]):
        return "snow"
    if any(k in fname for k in ["sand", "dust", "dusttornado"]):
        return "sand"
    if any(k in fname for k in ["night_", "night"]):
        return "night"
    w = extract_weather_from_aih_json(os.path.basename(img_path))
    return w


# ---------- geometry helpers ----------
def box_iou_xyxy(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def xyxy_to_xywh_norm(boxes, w_img, h_img):
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xc = ((boxes[:, 0] + boxes[:, 2]) * 0.5) / w_img
    yc = ((boxes[:, 1] + boxes[:, 3]) * 0.5) / h_img
    ww = (boxes[:, 2] - boxes[:, 0]) / w_img
    hh = (boxes[:, 3] - boxes[:, 1]) / h_img
    return np.stack([xc, yc, ww, hh], axis=1).astype(np.float32)


# ===========================================================
# MONTE CARLO DROPOUT(MC Dropout)
# ===========================================================
def preprocess_image(img_path, img_size=640, model=None):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, (img_size, img_size))

    img_rgb = img_resized[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    device = next(model.model.parameters()).device if model is not None else "cpu"
    return img_tensor.to(device), img


def enable_mc_dropout_only(module: torch.nn.Module):
    """
    전체 모델은 eval() 상태를 유지하되, Dropout 계층만 train()으로 전환한다.
    (ConvDropout 내부의 nn.Dropout 포함)
    """
    module.eval()  # 먼저 전체 eval (BatchNorm 등은 추론 경로 유지)
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()


def mc_dropout_single_pass(model, img_path, Tw=None):
    """
    - add_dropout_to_yolov8 로 forward 에 dropout이 이미 심어진 상태
    - model.model(img) 를 직접 사용
    - dropout은 F.dropout2d(..., training=True) 로 항상 활성화 됨
    Tw:
    - None 이면 BASE_CONF_THR를 사용 (테스트 코드 같은 기본 사용)
    - 값이 주어지면 weather-adaptive threshold 등으로 사용
    """
    # 1) 이미지를 텐서로 변환
    img_t, _ = preprocess_image(img_path, IMG_SIZE, model)
    if img_t is None:
        return (
            np.zeros((0, 4),  np.float32),
            np.zeros((0,   ), np.float32),
            np.zeros((0,   ), np.int32)
        )

    # 2) raw prediction + Non-Max Suppression
    with torch.no_grad():
        if Tw is None:
            Tw_eff = BASE_CONF_THR
        else:
            Tw_eff = float(Tw)

        conf_thres = max(BASE_CONF_THR, Tw_eff)

        raw = model.model(img_t)[0] # (1, N, 85)
        num_cls = raw.shape[-1] - 5
        pred = non_max_suppression(raw, conf_thres=conf_thres, iou_thres=0.6, max_det=300)[0]
        #results = model.predict(img_t, verbose=False)[0]  # Results 객체 1개

    # 3) 결과 없음 처리
    if pred is None or pred.size(0) == 0:
        return (
            np.zeros((0, 4),  np.float32),
            np.zeros((0,   ), np.float32),
            np.zeros((0,   ), np.int32)
        )

    boxes = pred[:, :4].cpu().numpy().astype(np.float32)
    confs = pred[:, 4].cpu().numpy().astype(np.float32)
    cls   = pred[:, 5].cpu().numpy().astype(np.int32)
    return boxes, confs, cls


def mc_dropout_inference(model, img_path, Tw=None, n_iter=MC_ITERS, T=TEMP):
    """
    이미지 단위 평균 confidence 및 표준편차 (Uw 계산용)
    Tw:
    - None 이면 BASE_CONF_THR 사용
    - 값이 주어지면 그대로 사용
    """
    enable_mc_dropout_only(model.model)
    vals = []
    for _ in range(n_iter):
        boxes, confs, classes = mc_dropout_single_pass(model, img_path, Tw)
        mean_conf = float(confs.mean()) if confs.size else 0.0
        vals.append(mean_conf)

    vals = np.array(vals, dtype=np.float32)
    model.model.eval()

    return float(vals.mean()), float(vals.std())


def mc_dropout_box_stats(model, img_path, base_boxes, base_classes, Tw, n_iter=MC_ITERS):
    """
    각 base box에 대해 Monte-Carlo 추론 결과를 IoU 기준으로 매칭하여
    base_boxes, base_classes 에 대해 MC-Dropout 기반 box-level uncertainty 계산.
    입력:
      - base_boxes: (M,4) np.ndarray (xyxy) — 기준 박스 (NMS 결과)
      - base_classes: (M,) np.ndarray — 각 박스의 클래스 ID
      - Tw: weather-adaptive threshold
    출력:
      - Umc_std: (M, ) 각 박스의 confidence 표준편차
      - BVAR: (M, ) 정규화된 bounding box(x_center,y_center,width,height) 분산의 평균
    """
    if base_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((len(base_boxes),), dtype=np.float32), np.ones((len(base_boxes),), dtype=np.float32)

    h_img, w_img = img.shape[:2]

    conf_samples = [[] for _ in range(len(base_boxes))]
    xywhn_samples = [[] for _ in range(len(base_boxes))]

    enable_mc_dropout_only(model.model)

    with torch.no_grad():
        for _ in range(n_iter):
            boxes, confs, classes = mc_dropout_single_pass(model, img_path, Tw)
            if boxes.size == 0:
                continue

            for i, (b, c_cls) in enumerate(zip(base_boxes, base_classes)):
                mask = (classes == c_cls)
                if not np.any(mask):
                    continue

                cand = boxes[mask]
                ious = box_iou_xyxy(b[None, :], cand)[0]
                j = int(np.argmax(ious))

                if ious[j] >= IOU_MATCH_THR:
                    sel_idx = np.nonzero(mask)[0][j]
                    conf_samples[i].append(float(confs[sel_idx]))
                    xywhn = xyxy_to_xywh_norm(boxes[sel_idx:sel_idx + 1], w_img, h_img)[0]
                    xywhn_samples[i].append(xywhn)
    model.model.eval()

    umc_std = np.ones((len(base_boxes),), dtype=np.float32)
    bvar = np.ones((len(base_boxes),), dtype=np.float32)
    for i in range(len(base_boxes)):
        if len(conf_samples[i]) > 1:
            umc_std[i] = np.std(np.array(conf_samples[i], dtype=np.float32))
        if len(xywhn_samples[i]) > 1:
            xs = np.stack(xywhn_samples[i], axis=0)  # (k,4)
            bvar[i] = xs.var(axis=0).mean().astype(np.float32)
    return umc_std, bvar


def mc_dropout_logits(model, img_path, Tw, n_iter=MC_ITERS):
    """
    Anchor level multi class MC 확률 분포 추정
    - YOLOv8 출력(raw)을 사용해 obj x cls 기반 joint prob(80 class + BG) 구성
    - 반환 : probs_mc (n_iter, N, num_cls+1), anchor_boxes_xyxy (N, 4)
    """
    img_t, _ = preprocess_image(img_path, IMG_SIZE, model)
    if img_t is None:
        return np.zeros((0,10), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    enable_mc_dropout_only(model.model)
    probs_list = []
    anchor_boxes_xyxy = None
    num_cls = None

    with torch.no_grad():
        for _ in range(n_iter):
            raw = model.model(img_t)[0]  # (1, N, 5+num_cls)
            raw = raw[0]  # (N, 5+num_cls)

            num_cls = raw.shape[1] - 5  # 4 box + obj + cls_num
            cls_logit = raw[:, 5:5 + num_cls]
            obj_logit = raw[:, 4]

            # 최초 1회에 anchor-level 박스 추출 (xywh -> xyxy)
            if anchor_boxes_xyxy is None:
                boxes_xywh = raw[:, :4].cpu().numpy().astype(np.float32)
                anchor_boxes_xyxy = xywh_to_xyxy(boxes_xywh)

            # sigmoid probs
            obj_prob = torch.sigmoid(obj_logit)  # (N,)
            cls_prob = torch.sigmoid(cls_logit)  # (N, num_cls)

            # joint prob for each class: p(class_k) = p(obj) * p(cls_k | obj)
            joint = obj_prob.unsqueeze(1) * cls_prob  # (N, num_cls)

            # background 확률 (남는 것)
            p_bg = torch.clamp(1.0 - joint.sum(dim=1, keepdim=True), min=1e-6)

            probs = torch.cat([p_bg, joint], dim=1)  # (N, num_cls+1)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)

            probs_list.append(probs.cpu().numpy())

    model.model.eval()

    if len(probs_list) == 0 or num_cls is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    probs_mc = np.stack(probs_list, axis=0)  # (n_iter, N, num_cls+1)
    return probs_mc, anchor_boxes_xyxy

def compute_multiclass_entropy(probs_mc: np.ndarray):
    """
    probs_mc: (n_iter, N ,num_cls+1) - MC-Dropout으로 얻은 확률 분포
    반환: anchor-level predictive entropy (N, )
    """
    if probs_mc.size == 0:
        return np.zeros((0, ), dtype=np.float32)

    #안전하게 clipping
    probs_mc = np.clip(probs_mc, 1e-9, 1.0)
    #predictive probability = E[p(y|x, w)] over MC samples
    mean_probs = probs_mc.mean(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)  # (N, )
    return entropy.astype(np.float32)


def match_entropy_to_boxes(base_boxes: np.ndarray, anchor_boxes: np.ndarray, ent_full: np.ndarray):
    """
    anchor-level entropy(ent_full) 를 box-level 로 매칭.
    - base_boxes: (M,4) NMS 결과 박스 (xyxy)
    - anchor_boxes: (N,4) anchor-level 박스 (xyxy)
    - ent_full: (N,) anchor-level entropy
    반환: (M,) box-level entropy
    """
    if base_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if (anchor_boxes is None or anchor_boxes.size == 0 or
        ent_full.size == 0 or anchor_boxes.shape[0] != ent_full.shape[0]):
        # anchor 정보를 못 얻으면 box-level entropy를 1.0 (최대 불확실)로 설정
        return np.ones((len(base_boxes),), dtype=np.float32)

    # IoU 기반 매칭
    ious = box_iou_xyxy(base_boxes, anchor_boxes)       # (M, N)
    max_idx = np.argmax(ious, axis=1)                   # (M,)
    max_iou = ious[np.arange(len(base_boxes)), max_idx] # (M,)

    ent = ent_full[max_idx]
    # 너무 IoU가 낮으면 anchor와 잘 매칭되지 않은 것으로 보고 entropy를 크게 설정
    ent[max_iou < 0.1] = 1.0
    return ent

# ===========================================================
# WEATHER-AWARE THRESHOLD
# ===========================================================
# weather별 평균 불확실성 계산(Uw) : 날씨별 평균 불확실성(Uw)에 따라 confidence threshold 조정
def compute_weather_adaptive_threshold(weather, Uw_dict, base_thr=BASE_CONF_THR):
    if weather in Uw_dict:
        Uw = Uw_dict[weather]
        return float(np.clip(base_thr + 0.3 * Uw, 0.1, 0.7))
    mapping = {
        "cloudy": 0.4,
        "rain": 0.55,
        "snow": 0.5,
        "fog": 0.6,
        "sand": 0.65,
        "night": 0.7,
        "clear": 0.3,
    }
    alpha = mapping.get(weather, 0.5)
    return float(np.clip(base_thr + 0.1 * (alpha - 0.5), 0.1, 0.7))


# ===========================================================
# MERGED DATASET (SHADOW) & YAML 생성
# ===========================================================
def _safe_symlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        # 파일시스템 정책상 심링크 불가 시 복사로 대체
        shutil.copy2(src, dst)


def _expand_images_list(images_dir):
    return sorted(glob.glob(os.path.join(images_dir, "*.jpg"))) + \
           sorted(glob.glob(os.path.join(images_dir, "*.png")))


def build_shadow_merged_dataset(sd_yaml_path, td_images_dir, td_pseudo_label_dir, out_root):
    """
    out_root/images/train : SD train imgs + TD train imgs (symlink)
    out_root/labels/train : SD train labels + TD pseudo labels (symlink)
    """
    with open(sd_yaml_path, "r") as f:
        sd_cfg = yaml.safe_load(f)
    sd_train_images_dir = sd_cfg['train']  # SD의 images/train 디렉터리
    sd_train_labels_dir = sd_train_images_dir.replace("/images/", "/labels/")

    # 출력 디렉터리
    img_out = Path(out_root) / "images" / "train"
    lbl_out = Path(out_root) / "labels" / "train"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # 1) SD train (이미지/라벨) 심링크
    for img in _expand_images_list(sd_train_images_dir):
        stem = Path(img).stem
        lbl = os.path.join(sd_train_labels_dir, f"{stem}.txt")
        if os.path.exists(lbl):
            _safe_symlink(img, str(img_out / f"{stem}{Path(img).suffix}"))
            _safe_symlink(lbl, str(lbl_out / f"{stem}.txt"))

    # 2) TD train (이미지) + TD pseudo-label (라벨) 심링크
    for img in _expand_images_list(td_images_dir):
        weather = get_weather_label(img)
        if weather in ["clear", "unknown"]:
            continue

        stem = Path(img).stem
        pseudo = os.path.join(td_pseudo_label_dir, f"{stem}.txt")
        # pseudo label이 없더라도 빈 파일을 만들어둬 YOLO dataloader 에러 방지
        if not os.path.exists(pseudo):
            os.makedirs(td_pseudo_label_dir, exist_ok=True)
            open(pseudo, "a").close()
        _safe_symlink(img,   str(img_out / f"{stem}{Path(img).suffix}"))
        _safe_symlink(pseudo, str(lbl_out / f"{stem}.txt"))

    print(f"[INFO] Shadow merged dataset built at: {out_root}")


def write_merged_yaml_from_shadow(sd_yaml_path, shadow_root, merged_yaml_path):
    with open(sd_yaml_path, "r") as f:
        sd_cfg = yaml.safe_load(f)

    merged_cfg = sd_cfg.copy()
    # shadow 데이터셋의 images/train 경로만 지정하면, YOLO가 자동으로 labels/train을 매칭
    merged_cfg['train'] = str(Path(shadow_root) / "images" / "train")
    # val/test 는 기존 SD/TD YAML 정책 유지 (필요 시 수정 가능)
    merged_cfg['val']   = sd_cfg.get('val', None)
    merged_cfg['test']  = sd_cfg.get('test', None)

    with open(merged_yaml_path, "w") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)
    print(f"[INFO] MERGED YAML saved at {merged_yaml_path}")


# ===========================================================
# TRAIN / EVAL
# ===========================================================
def train_source_model():
    model = YOLO('yolov8l.pt')
    model = add_dropout_to_yolov8(model)  # Dropout 삽입 (Option A)
    model.train(
        data=SD_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        project=PROJECT,
        name="sd_train",
        exist_ok=True
    )
    return f"{SD_WEIGHT_DIR}/best.pt", f"{SD_WEIGHT_DIR}/last.pt"


def evaluate_yolo_metrics(model_path, data_yaml):
    model = YOLO(model_path)    # dropout 없이 original model 검증(dropout patch 미적용해야 한다.)
    # model = add_dropout_to_yolov8(model)  # 평가 모델에도 동일 구조 삽입 (일관성 유지)
    res = model.val(data=data_yaml, imgsz=IMG_SIZE, batch=BATCH, verbose=True)

    f1_attr = getattr(res.box, 'f1', 0.0)
    if isinstance(f1_attr, (list, np.ndarray)):
        f1_value = float(np.mean(f1_attr))
    else:
        f1_value = float(f1_attr)

    metrics = {
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
        "Precision": float(res.box.mp),
        "Recall": float(res.box.mr),
        "F1": f1_value
    }
    return metrics


def domain_transfer_degradation_rate(source_score: float, target_score: float) -> float:
    return (source_score - target_score) / max(source_score, 1e-6) * 100.0


def robustness_index(scores):
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return 0.0
    mean, std = arr.mean(), arr.std()
    if mean == 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (std / mean))))


def evaluate_cross_domain(model_path, sd_yaml, td_yaml=TD_YAML):
    sd_val_metrics = evaluate_yolo_metrics(model_path, sd_yaml)
    td_test_metrics = evaluate_yolo_metrics(model_path, td_yaml)
    print("[CrossEval] SD Validation:", sd_val_metrics)
    print("[CrossEval] TD Test:", td_test_metrics)
    dt_rate = domain_transfer_degradation_rate(sd_val_metrics['mAP50'], td_test_metrics['mAP50'])
    robustness = robustness_index([sd_val_metrics['mAP50'], td_test_metrics['mAP50']])
    print(f"[CrossEval] DTDR: {dt_rate:.2f}%, RI: {robustness:.4f}")
    summary = {
        "SD mAP50": sd_val_metrics['mAP50'],
        "TD mAP50": td_test_metrics['mAP50'],
        "Precision": td_test_metrics['Precision'],
        "Recall": td_test_metrics['Recall'],
        "DTDR(%)": dt_rate,
        "Robustness Index": robustness
    }
    summary_path = Path(PROJECT) / "summary_cross_domain.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[INFO] Cross-domain summary saved at {summary_path}")
    return summary


# ===========================================================
# PSEUDO-LABEL
# ===========================================================
def compute_weather_uncertainty(td_img_dir, model, max_per_weather=None):
    all_imgs = sorted(glob.glob(os.path.join(td_img_dir, "*.jpg")))
    all_imgs = [p for p in all_imgs if get_weather_label(p) not in ["clear", "unknown"]]

    # 1) 날씨별로 먼저 그룹핑
    weather_to_imgs = {}
    for p in all_imgs:
        w = get_weather_label(p)
        weather_to_imgs.setdefault(w, []).append(p)

    # 2) max_per_weather가 None이면 "최소 개수"로 자동 설정
    if max_per_weather is None:
        min_cnt = min(len(v) for v in weather_to_imgs.values())
        max_per_weather = min_cnt  # 지금 케이스에서는 250이 될 것

    # 3) 날씨별로 최대 max_per_weather장씩만 샘플링
    img_list = []
    for w, imgs in weather_to_imgs.items():
        if len(imgs) > max_per_weather:
            imgs = random.sample(imgs, max_per_weather)  # 또는 random.sample(imgs, max_per_weather)
        img_list.extend(imgs)

    print(f"[Uw] Using up to {max_per_weather} images per weather, total {len(img_list)}")

    # 이후는 동일하게 std_conf 축적
    weather_dict = {}
    enable_mc_dropout_only(model.model)
    for img_path in tqdm(img_list, desc="Uw"):
        weather = get_weather_label(img_path)
        _, std_conf = mc_dropout_inference(model, img_path, BASE_CONF_THR)
        if not np.isnan(std_conf):
            weather_dict.setdefault(weather, []).append(std_conf)

    Uw_dict = {w: np.mean(stds) for w, stds in weather_dict.items() if len(stds) > 0}
    log_str = {w: f"{Uw_dict[w]:.4f} (N={len(weather_dict[w])})" for w in Uw_dict}
    print("[Uw] mean uncertainty by weather:", log_str)
    return Uw_dict


def generate_td_pseudo_labels(model_path, td_img_dir, txt_dir, vis_dir):
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 가중치 로딩 + Dropout 삽입
    model = YOLO(model_path)
    model = add_dropout_to_yolov8(model, p_backbone=0.1, p_neck=0.1, p_head=0.3)

    # 날씨별 평균 불확실성
    Uw_dict = compute_weather_uncertainty(td_img_dir, model)

    img_list = sorted(glob.glob(os.path.join(td_img_dir, "*.jpg")))

    img_list = [p for p in img_list if get_weather_label(p) not in ["clear", "unknown"]]

    print(f"[INFO] TD pseudo-label generation target images: {len(img_list)}")

    for img_path in tqdm(img_list, desc="Pseudo-label"):
        weather = get_weather_label(img_path)

        # Night / Fog / Sand → conf 낮아짐 → Thr 낮춰줘야 detection 유지됨
        # weather_boost = {
        #    "night": -0.10,
        #    "fog": -0.07,
        #    "rain": -0.04,
        #    "snow": -0.05,
        #    "sand": -0.08,
        #    "cloudy": -0.02,
        # }

        # Uw 기반 adaptive threshold
        Tw = compute_weather_adaptive_threshold(weather, Uw_dict)

        # base detection
        boxes, confs, classes = mc_dropout_single_pass(model, img_path, Tw)

        # SD→TD 클래스 맵핑
        keep = [i for i, c in enumerate(classes) if int(c) in SD_TO_TD_IDX]
        if len(keep) == 0:
            open(os.path.join(txt_dir, f"{Path(img_path).stem}.txt"), "a").close()
            # 빈 시각화도 저장(요구 시) — 여기선 저장 포맷 유지 위해 그대로 저장
            vis_save = os.path.join(vis_dir, f"{Path(img_path).stem}_vis.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(vis_save, img)
            continue

        boxes = boxes[keep]
        confs = confs[keep]
        classes = np.array([SD_TO_TD_IDX[int(classes[i])] for i in keep], dtype=np.int32)

        # 박스 단위 불확실성(umc_std, BVAR)
        umc_std, bvar = mc_dropout_box_stats(model, img_path, boxes, classes, Tw, MC_ITERS)

        probs_mc, anchor_boxes = mc_dropout_logits(model, img_path, Tw, MC_ITERS)
        ent_full = compute_multiclass_entropy(probs_mc)
        ent = match_entropy_to_boxes(boxes, anchor_boxes, ent_full)

        # loose filtering
        loose_mask = (confs >= Tw*0.85) & (ent <= ENTROPY_THR*1.1)

        # strong filtering
        strong_mask = loose_mask & (umc_std <= UMC_STD_THR) & (bvar <= BVAR_THR)
        boxes_f, confs_f, classes_f = boxes[strong_mask], confs[strong_mask], classes[strong_mask]

        # YOLO txt 저장
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bw = (boxes_f[:, 2] - boxes_f[:, 0]) / w
        bh = (boxes_f[:, 3] - boxes_f[:, 1]) / h
        valid_mask = (bw >= 0.01) & (bh >= 0.01)

        boxes_f = boxes_f[valid_mask]
        confs_f = confs_f[valid_mask]
        classes_f = classes_f[valid_mask]

        txt_path = os.path.join(txt_dir, f"{Path(img_path).stem}.txt")
        with open(txt_path, "w") as f:
            for b, c in zip(boxes_f, classes_f):
                x1, y1, x2, y2 = b
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if bw < 0.01 or bh < 0.01:  # FIX4: invalid box 제거
                    continue

                f.write(
                    f"{int(c)} {(b[0] + b[2]) / (2 * w):.6f} {(b[1] + b[3]) / (2 * h):.6f} "
                    f"{bw:.6f} {bh:.6f}\n"
                )

        # 오버레이 저장 (Target: train 7000 + test 1000 → 총 8000장)
        vis_save = os.path.join(vis_dir, f"{Path(img_path).stem}_vis.jpg")
        img = cv2.imread(img_path)
        if img is not None:
            for box, conf, cls_id in zip(boxes_f, confs_f, classes_f):
                color = (0, 255, 0) if conf >= Tw else (0, 0, 255)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{int(cls_id)}:{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            cv2.imwrite(vis_save, img)
    log_pseudo_stats(txt_dir)


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    print("========== UDA Self-Training Full Pipeline ==========")

    # 1. Train source-only model (필요 시 주석 해제)
    print("[Step1] Train Source-only model")
    best_weight, _ = train_source_model()

    # 2. Evaluate SD val and TD test with the source model (cross-domain) (필요 시 주석 해제)
    print("[Step2] Evaluate Source-only model (SD val -> TD test)")
    evaluate_cross_domain(best_weight, SD_YAML, TD_YAML)

    # 이미 학습된 베스트 가중치 사용 가능
    # best_weight = f"{SD_WEIGHT_DIR}/best.pt"

    # 3. Generate TD pseudo-labels (WP-PL-UC) on TD train
    print("[Step3a] Generate TD pseudo-labels (WP-PL-UC) - TRAIN (and overlays)")
    generate_td_pseudo_labels(best_weight, TD_IMAGES_DIR, TD_PSEUDO_LABEL_DIR, TD_PSEUDO_VIS_DIR)

    # 4. Generate overlays (and txt) on TD TEST (1,000 images)
    print("[Step3b] Generate TD pseudo-labels (WP-PL-UC) - TEST (and overlays)")
    generate_td_pseudo_labels(best_weight, TD_TEST_IMAGES_DIR, TD_PSEUDO_LABEL_TEST_DIR, TD_PSEUDO_VIS_TEST_DIR)

    # 5. Build shadow merged dataset & write YAML (SD + TD pseudo)
    print("[Step4] Build shadow MERGED dataset & YAML")
    build_shadow_merged_dataset(SD_YAML, TD_IMAGES_DIR, TD_PSEUDO_LABEL_DIR, MERGED_DATASET_DIR)
    write_merged_yaml_from_shadow(SD_YAML, MERGED_DATASET_DIR, MERGED_YAML)

    # 6. Train on merged dataset (fine-tune)
    print("[Step5] Train on MERGED dataset")
    merged = YOLO('yolov8l.pt')
    merged = add_dropout_to_yolov8(merged)  # fine-tuning 모델에도 동일 삽입
    merged.train(
        data=MERGED_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        project=PROJECT,
        name="merged_train",
        exist_ok=True
    )

    # 7. Evaluate merged model cross-domain
    print("[Step6] Evaluate merged model (SD val -> TD test)")
    evaluate_cross_domain(f"{PROJECT}/merged_train/weights/best.pt", SD_YAML, TD_YAML)

    print("========== Pipeline Completed ==========")
