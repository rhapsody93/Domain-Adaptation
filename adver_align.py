import os
import json
import datetime
import yaml
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from typing import List
from tqdm import tqdm

# ===========================================================
# CONFIGURATION
# ===========================================================
PROJECT = "/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/domain_adaptation/adversarial_alignment"
BASE_DIR = Path(PROJECT)
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SD_YAML = "/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/source_domain/train/merged_dataset.yaml"
TD_YAML = "/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/target_domain/train/merged_dataset.yaml"

IMG_SIZE = 640
BATCH = 8
EPOCHS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AA_BATCH_SIZE = 4
AA_EPOCHS = 5           # v6-strong: 약간 더 길게
LR = 5e-5               # v6-strong: 한 단계 낮춰 안정화
LAMBDA_ADV = 0.5        # v6-strong: GRL 강도 증가
NUM_WORKERS = 0

TD_OVERLAY_CONF = 0.15  # target domain overlay를 위한 confidence threshold

# ===========================================================
# YAML & CLASS MAPPING
# ===========================================================
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    return d.get("names", [])


def check_class_mapping():
    global SD_CLASSES, TD_CLASSES, SD_TO_TD_IDX

    SD_CLASSES = load_classes_from_yaml(SD_YAML)
    TD_CLASSES = load_classes_from_yaml(TD_YAML)

    COMMON = [c for c in SD_CLASSES if c in TD_CLASSES]

    SD_TO_TD_IDX = {
        SD_CLASSES.index(c): TD_CLASSES.index(c)
        for c in COMMON
    }

    print("\n===== [CHECK] CLASS MAPPING =====")
    print("SD_CLASSES:", SD_CLASSES)
    print("TD_CLASSES:", TD_CLASSES)
    print("COMMON_CLASSES:", COMMON)
    print("SD_TO_TD_IDX:", SD_TO_TD_IDX)
    print("=================================\n")


# ===========================================================
# FEATURE EXTRACTION DATASET
# ===========================================================
class ImageDataset(Dataset):
    def __init__(self, img_paths: List[str], img_size=IMG_SIZE):
        self.paths = img_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img_tensor, path


# ===========================================================
# GRADIENT REVERSAL LAYER
# ===========================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# ===========================================================
# DOMAIN DISCRIMINATOR
# ===========================================================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================
# UTILS
# ===========================================================
def load_image_paths(yaml_path, split="train") -> List[str]:
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    split_dir = d.get(split, None)
    if split_dir is None:
        raise ValueError(f"No '{split}' field in {yaml_path}")
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(os.path.dirname(yaml_path), split_dir)
    imgs = list(Path(split_dir).rglob("*.jpg")) + list(Path(split_dir).rglob("*.png"))
    return [str(p) for p in imgs]


def get_split_dir_from_yaml(yaml_path: str, split: str) -> str:
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    split_dir = d.get(split, None)
    if split_dir is None:
        raise ValueError(f"No '{split}' field in {yaml_path}")
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(os.path.dirname(yaml_path), split_dir)
    return split_dir


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


# ===========================================================
# TRAINER (v7-strong)
#   - YOLOv8 DetectionModel + forward hook 기반 backbone feature
#   - detection head freeze + backbone only optimizer
#   - adversarial GRL only (det loss OFF)
#   - TD test overlay: manual predict + OpenCV
# ===========================================================
class AdversarialTrainer:
    def __init__(self, sd_yaml, td_yaml, device=DEVICE):
        self.sd_yaml = sd_yaml
        self.td_yaml = td_yaml
        self.device = device

        # Run dirs
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_DIR / f"exp_{self.run_id}"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.results_dir = self.run_dir / "results"
        self.losses_dir = self.results_dir / "losses"

        for d in [
            self.run_dir, self.checkpoints_dir,
            self.metrics_dir, self.results_dir, self.losses_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Loss trackers
        self.task_losses = []         # det loss는 0으로만 기록(그래프용)
        self.domain_losses = []
        self.adversarial_losses = []

        # ------------------------------------------
        # YOLO Load (pretrained source-only)
        # ------------------------------------------
        PRETRAINED_SD_WEIGHT = (
            "/home/jhkim/anaconda3/envs/project_ms/custom_datasets/"
            "Object_Detection/related_codes/domain_adaptation/self_training/"
            "sd_train/weights/best.pt"
        )
        print(f"[INFO] Loading SD pretrained model: {PRETRAINED_SD_WEIGHT}")

        # self.model: Ultralytics YOLO wrapper
        # self.det_model: 내부 DetectionModel
        self.model = YOLO(PRETRAINED_SD_WEIGHT)
        self.det_model = self.model.model.to(self.device)

        # ------------------------------------------
        # Backbone / Head 분리 (optimizer만 분리)
        # ------------------------------------------
        modules = list(self.det_model.model)
        self.backbone_layers = modules[:-1]
        self.head = modules[-1]

        # Head freeze (detection head는 업데이트 X)
        for p in self.head.parameters():
            p.requires_grad = False

        # Backbone 파라미터만 optimizer에 등록
        backbone_params = []
        for m in self.backbone_layers:
            backbone_params += list(m.parameters())
        self.opt_backbone = optim.Adam(backbone_params, lr=LR)

        # ------------------------------------------
        # Feature hook (graph 그대로, 중간 feature 추출)
        # ------------------------------------------
        self.backbone_feats = None
        self._register_feature_hook()
        self._warmup()

        # ------------------------------------------
        # Datasets
        # ------------------------------------------
        self.sd_paths = load_image_paths(sd_yaml, split="train")
        self.td_paths = load_image_paths(td_yaml, split="train")

        self.sd_loader = DataLoader(
            ImageDataset(self.sd_paths), batch_size=AA_BATCH_SIZE,
            shuffle=True, num_workers=NUM_WORKERS
        )
        self.td_loader = DataLoader(
            ImageDataset(self.td_paths), batch_size=AA_BATCH_SIZE,
            shuffle=True, num_workers=NUM_WORKERS
        )

        # ------------------------------------------
        # Feature dim 자동 추론 + Discriminator 설정
        # ------------------------------------------
        feat_dim = self._infer_featdim()
        print(f"[INFO] Feature dim = {feat_dim}")

        self.discriminator = DomainDiscriminator(feat_dim).to(self.device)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=LR)
        self.criterion_adv = nn.BCELoss()

    # ===========================================================
    # Feature hook registration
    # ===========================================================
    def _register_feature_hook(self):
        """
        det_model.model의 깊은 레이어에 hook을 걸어서
        backbone feature를 추출한다.
        """
        def hook_fn(module, inp, out):
            if isinstance(out, (list, tuple)):
                outs = [o for o in out if isinstance(o, torch.Tensor)]
                if len(outs) == 0:
                    return
                self.backbone_feats = outs
            else:
                self.backbone_feats = out

        seq = self.det_model.model
        candidates = [-5, -6, -7]

        for idx in candidates:
            try:
                seq[idx].register_forward_hook(hook_fn)
                print(f"[INFO] Hook registered at det_model.model[{idx}]")
                return
            except Exception:
                continue

        seq[-3].register_forward_hook(hook_fn)
        print("[WARN] Using fallback hook at det_model.model[-3]")

    # ===========================================================
    # Warmup for hook
    # ===========================================================
    def _warmup(self):
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        with torch.no_grad():
            _ = self.det_model(dummy)

    # ===========================================================
    # Backbone feature (hook 기반)
    # ===========================================================
    def forward_backbone(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        det_model 전체 forward를 사용하되,
        중간 레이어 hook에서 backbone_feats를 받아와
        GAP + concat으로 feature vector를 만든다.
        """
        self.backbone_feats = None
        _ = self.det_model(imgs)  # hook에서 backbone_feats 세팅

        feats = self.backbone_feats
        if feats is None:
            return torch.zeros(imgs.size(0), 512, device=self.device)

        if isinstance(feats, (list, tuple)):
            pooled = []
            for f in feats:
                if not isinstance(f, torch.Tensor):
                    continue
                if f.ndim == 4:
                    pooled.append(f.mean(dim=(2, 3)))  # GAP
                else:
                    pooled.append(f)
            if len(pooled) == 0:
                return torch.zeros(imgs.size(0), 512, device=self.device)
            feats_vec = torch.cat(pooled, dim=1)
        else:
            feats_vec = feats
            if isinstance(feats_vec, torch.Tensor) and feats_vec.ndim == 4:
                feats_vec = feats_vec.mean(dim=(2, 3))

        return feats_vec

    # ===========================================================
    # Feature dim infer (옵션 A: 자동 추론)
    # ===========================================================
    def _infer_featdim(self) -> int:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=self.device)
        with torch.no_grad():
            feats = self.forward_backbone(dummy)
        return int(feats.shape[1])

    # ===========================================================
    # ADVERSARIAL EPOCH (v7-strong: det loss 제거, feature-level alignment만)
    # ===========================================================
    def adversarial_epoch(self, epochs=AA_EPOCHS):
        print(f"[INFO] Adversarial alignment (v7-strong) — {epochs} epochs (det loss OFF)")

        self.det_model.train()
        self.discriminator.train()

        for ep in range(epochs):

            sd_iter = iter(self.sd_loader)
            td_iter = iter(self.td_loader)

            steps = min(len(self.sd_loader), len(self.td_loader))

            disc_losses = []
            adv_losses = []

            for s in range(steps):

                # ---------------------------------------------------------
                # 1) Load batches (SD + TD)
                # ---------------------------------------------------------
                try:
                    sd_imgs, _ = next(sd_iter)
                except StopIteration:
                    sd_iter = iter(self.sd_loader)
                    sd_imgs, _ = next(sd_iter)

                try:
                    td_imgs, _ = next(td_iter)
                except StopIteration:
                    td_iter = iter(self.td_loader)
                    td_imgs, _ = next(td_iter)

                sd_imgs = sd_imgs.to(self.device)
                td_imgs = td_imgs.to(self.device)

                bs_sd = sd_imgs.size(0)
                bs_td = td_imgs.size(0)

                # ---------------------------------------------------------
                # 2) Step 1: Discriminator 업데이트 (backbone grad X)
                #    - SD, TD feature는 detach해서 D만 학습
                #    - label smoothing (0.9 / 0.1)
                # ---------------------------------------------------------
                with torch.no_grad():
                    feats_sd_det = self.forward_backbone(sd_imgs)
                    feats_td_det = self.forward_backbone(td_imgs)

                self.opt_disc.zero_grad()

                feats_det = torch.cat([feats_sd_det, feats_td_det], dim=0)
                labels_sd = torch.full((bs_sd, 1), 0.9, device=self.device)
                labels_td = torch.full((bs_td, 1), 0.1, device=self.device)
                labels_det = torch.cat([labels_sd, labels_td], dim=0)

                preds_disc = self.discriminator(feats_det)
                loss_d = self.criterion_adv(preds_disc, labels_det)
                loss_d.backward()
                self.opt_disc.step()

                disc_losses.append(loss_d.item())

                # ---------------------------------------------------------
                # 3) Step 2: Backbone + GRL (domain confusion)
                #    - backbone에는 grad 허용, detection loss는 사용하지 않음
                #    - GRL 강도 LAMBDA_ADV 사용
                #    - Discriminator 파라미터의 requires_grad는 건드리지 않고,
                #      opt_disc.step()만 호출하지 않아서 D는 업데이트 안 됨
                # ---------------------------------------------------------
                self.opt_backbone.zero_grad()

                feats_sd_g = self.forward_backbone(sd_imgs)      # grad O
                feats_td_g = self.forward_backbone(td_imgs)      # grad O

                feats_concat = torch.cat([feats_sd_g, feats_td_g], dim=0)
                feats_adv = grad_reverse(feats_concat, lambda_=LAMBDA_ADV)

                labels_sd_adv = torch.full((bs_sd, 1), 0.1, device=self.device)
                labels_td_adv = torch.full((bs_td, 1), 0.9, device=self.device)
                labels_adv = torch.cat([labels_sd_adv, labels_td_adv], dim=0)

                preds_adv = self.discriminator(feats_adv)
                loss_adv = self.criterion_adv(preds_adv, labels_adv)

                # 여기서 그래프가 backbone + D 양쪽으로 열려 있어도
                # opt_backbone.step()만 호출하므로 실제 업데이트는 backbone만 됨
                loss_adv.backward()
                self.opt_backbone.step()

                adv_losses.append(loss_adv.item())

                # 그래프 기록용
                self.task_losses.append(0.0)
                self.domain_losses.append(loss_d.item())
                self.adversarial_losses.append(loss_adv.item())

            print(f"[Ep {ep+1}/{epochs}] "
                  f"D: {np.mean(disc_losses):.4f}, "
                  f"Adv: {np.mean(adv_losses):.4f}")

        return

    # ===========================================================
    # Loss curves 저장
    # ===========================================================
    def plot_losses(self):
        iters_task = range(1, len(self.task_losses) + 1)
        plt.figure(figsize=(8, 6))
        if len(self.task_losses):
            plt.plot(iters_task, self.task_losses, label="task_loss(det=0)")
        if len(self.domain_losses):
            plt.plot(range(1, len(self.domain_losses) + 1),
                     self.domain_losses, label="domain_loss(D)")
        if len(self.adversarial_losses):
            plt.plot(range(1, len(self.adversarial_losses) + 1),
                     self.adversarial_losses, label="adv_loss(GRL)")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = self.losses_dir / "losses_plot.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Loss curves saved at {out_path}")

    # ===========================================================
    # Split metrics bar plot 저장
    # ===========================================================
    def _plot_split_metrics(self, metrics: dict, split_name: str):
        keys = ["precision", "recall", "f1", "mAP50", "mAP50-95"]
        disp_keys = [k for k in keys if k in metrics]
        vals = [metrics[k] for k in disp_keys]

        plt.figure(figsize=(6, 4))
        plt.bar(disp_keys, vals)
        plt.ylim(0.0, 1.0)
        plt.title(f"Metrics ({split_name})")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight.tight_layout = True
        plt.tight_layout()

        out_dir = self.metrics_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics_bar.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Metrics bar plot saved at {out_path}")

    # ===========================================================
    # Evaluation (val/test) - YOLO .val()만 메트릭 계산용으로 사용
    # ===========================================================
    def evaluate(self, yaml_file: str, split_name="val", imgsz=IMG_SIZE):
        """
        - YOLOv8 .val() 사용 (메트릭 계산)
        - overlay 이미지는 test split의 경우 별도 함수에서 직접 저장
        """
        overlay_save = False  # v7-strong: .val()에서는 overlay 저장 안 함
        print(f"[INFO] Evaluating {yaml_file} [{split_name}] (save overlays in .val: {overlay_save})")

        try:
            res = self.model.val(
                data=yaml_file,
                split=split_name,
                imgsz=imgsz,
                device=self.device,
                save=overlay_save,
                save_txt=True,
                plots=True,
                project=str(self.results_dir / f"{split_name}_eval"),
                exist_ok=True,
            )
        except Exception as e:
            print(f"[WARN] model.val failed: {e}")
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }
            return metrics

        try:
            precision = float(res.box.mp)
            recall = float(res.box.mr)
            mAP50 = float(res.box.map50)
            mAP50_95 = float(res.box.map)
            f1_attr = getattr(res.box, "f1", 0.0)
            if isinstance(f1_attr, (list, np.ndarray)):
                f1 = float(np.mean(f1_attr))
            else:
                f1 = float(f1_attr)
        except Exception:
            metrics_dict = getattr(res, "metrics", None)
            if metrics_dict is None and hasattr(res, "results_dict"):
                metrics_dict = res.results_dict
            if metrics_dict is None:
                metrics_dict = {}
            precision = float(metrics_dict.get("precision(B)", metrics_dict.get("precision", 0.0)))
            recall = float(metrics_dict.get("recall(B)", metrics_dict.get("recall", 0.0)))
            mAP50 = float(metrics_dict.get("mAP50(B)", metrics_dict.get("map50", 0.0)))
            mAP50_95 = float(metrics_dict.get("mAP50-95(B)", metrics_dict.get("map", 0.0)))
            f1 = (
                2 * precision * recall / (precision + recall + 1e-8)
                if (precision + recall) > 0
                else 0.0
            )

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
        }

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.metrics_dir / f"{split_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"[INFO] Metrics saved at {metrics_path}")

        self._plot_split_metrics(metrics, split_name)

        return metrics

    # ===========================================================
    # Target Domain test overlay 저장 (predict + OpenCV 수동 저장)
    # ===========================================================
    def save_target_overlays(self, conf: float = TD_OVERLAY_CONF, max_imgs: int = None):
        """
        TD test split에 대해 YOLOv8 predict()를 호출하고,
        결과를 OpenCV로 그려서 overlay 이미지를 직접 저장한다.
        """
        print("[INFO] Saving target domain test overlays (manual predict + OpenCV) ...")

        test_dir = get_split_dir_from_yaml(self.td_yaml, "test")
        img_paths = sorted(list(Path(test_dir).rglob("*.jpg")) +
                           list(Path(test_dir).rglob("*.png")))

        if max_imgs is not None:
            img_paths = img_paths[:max_imgs]

        out_dir = self.results_dir / "test_overlays"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.model.eval()

        for img_path in tqdm(img_paths, desc="TD overlay"):
            im0 = cv2.imread(str(img_path))
            if im0 is None:
                continue

            results = self.model(
                im0,
                imgsz=IMG_SIZE,
                conf=conf,
                verbose=False
            )
            if len(results) == 0:
                overlay = im0
            else:
                res = results[0]
                overlay = res.plot()  # BGR numpy

            save_path = out_dir / Path(img_path).name
            cv2.imwrite(str(save_path), overlay)

        print(f"[INFO] Target domain overlay images saved under: {out_dir}")

    # ===========================================================
    # Cross-domain 평가 (DTDR & RI)
    # ===========================================================
    def evaluate_cross_domain(self, sd_metrics: dict, td_metrics: dict):
        sd_mAP50 = sd_metrics["mAP50"]
        td_mAP50 = td_metrics["mAP50"]
        dt_rate = domain_transfer_degradation_rate(sd_mAP50, td_mAP50)
        ri = robustness_index([sd_mAP50, td_mAP50])

        summary = {
            "SD_val": sd_metrics,
            "TD_test": td_metrics,
            "DTDR(%)": dt_rate,
            "Robustness Index": ri,
        }
        summary_path = self.run_dir / "summary_cross_domain.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[INFO] Cross-domain summary saved at {summary_path}")
        print(f"[CrossEval] DTDR: {dt_rate:.2f}%, RI: {ri:.4f}")
        return summary

    # ===========================================================
    # Full pipeline
    # ===========================================================
    def run(self, epochs=EPOCHS):
        print("[STAGE 1] Skip pretraining — using existing source-only best.pt")

        print("[STAGE 2] Adversarial alignment (Domain Discriminator training, v7-strong)...")
        self.adversarial_epoch(epochs=AA_EPOCHS)

        self.plot_losses()

        print("[STAGE 3] Evaluate on Source Domain (val)...")
        val_metrics = self.evaluate(self.sd_yaml, "val")

        print("[STAGE 4] Evaluate on Target Domain (test, manual overlays)...")
        test_metrics = self.evaluate(self.td_yaml, "test")

        self.save_target_overlays(conf=TD_OVERLAY_CONF)

        print("[STAGE 5] Cross-domain metrics (DTDR, RI)...")
        summary = self.evaluate_cross_domain(val_metrics, test_metrics)

        print("\n========== FINAL SUMMARY ==========")
        print(json.dumps(summary, indent=4))
        print("===================================")

        return summary


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    check_class_mapping()
    print("========== YOLOv8 Adversarial Alignment v7-strong (Final) ==========")
    trainer = AdversarialTrainer(SD_YAML, TD_YAML)
    summary = trainer.run(epochs=EPOCHS)
    print("[INFO] Training and evaluation complete.")
