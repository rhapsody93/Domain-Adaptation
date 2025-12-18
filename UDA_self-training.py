import subprocess, os, shutil, time, csv
from pathlib import Path
import yaml

# ---------------- CONFIG ----------------
SD_YAML = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/source_domain/train/merged_dataset.yaml'
SD_IMAGES_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/source_domain_weather/MERGED_SD_BB/images/train'
SD_LABELS_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/source_domain_weather/MERGED_SD_BB/labels/train'

TD_IMAGES_DIR = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/target_domain_weather/MERGED_TD_BB/images/train'
PROJECT = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/domain_adaptation/self_training'
TD_PSEUDO_LABEL_DIR = f"{PROJECT}/pseudo_labels/td_train_txt"
TD_PSEUDO_VIS_DIR = f"{PROJECT}/pseudo_labels/td_train_vis"

SO_MODEL = '/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/cross_domain/sd_train/weights/best.pt'

INITIAL_CONF = 0.6
MIN_CONF = 0.1
CONF_STEP = 0.1

IMG_SIZE = 640
BATCH = 16

MERGED_DATASET_DIR = f"{PROJECT}/merged_sd_pseudo"
MERGED_YAML = f"{PROJECT}/merged_with_pseudo.yaml"

# ---------------- Helpers ----------------
def read_sd_traffic_sign_index(sd_yaml_path, traffic_name_candidates=("traffic sign","traffic_sign","traffic-sign")):
    try:
        with open(sd_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", None)
        if names is None:
            return None
        if isinstance(names, dict):
            for k, v in names.items():
                if any(cand.lower() == str(v).lower() for cand in traffic_name_candidates):
                    return int(k)
            return None
        if isinstance(names, list):
            for i, v in enumerate(names):
                if any(cand.lower() == str(v).lower() for cand in traffic_name_candidates):
                    return i
            return None
        return None
    except Exception as e:
        print("Error reading SD YAML for names:", e)
        return None

def find_latest_pseudo_dirs(project_dir: Path, name_substr="pseudo_infer", wait_seconds=1):
    cand = []
    project_dir = Path(project_dir)
    time.sleep(wait_seconds)
    for d in project_dir.iterdir():
        if d.is_dir() and name_substr in d.name:
            cand.append(d)
    detect_dir = project_dir / "runs" / "detect"
    if detect_dir.exists():
        for d in detect_dir.iterdir():
            if d.is_dir() and name_substr in d.name:
                cand.append(d)
    for base in list(cand):
        for sub in base.iterdir():
            if sub.is_dir() and name_substr in sub.name:
                cand.append(sub)
    cand_unique = {p.resolve(): p for p in cand}.values()
    return sorted(cand_unique, key=lambda p: p.stat().st_mtime, reverse=True)

def collect_label_txts_from_dir(base_dir: Path):
    files = [p for p in Path(base_dir).rglob("*.txt")]
    labels_candidates = []
    if (Path(base_dir) / "labels").exists():
        labels_candidates.append(Path(base_dir) / "labels")
    for d in Path(base_dir).glob("labels*"):
        if d.is_dir():
            labels_candidates.append(d)
    for d in labels_candidates:
        files.extend(d.rglob("*.txt"))
    return list({f.resolve(): f for f in files}.values())

# ---------------- CORE: robust pseudo inference ----------------
def run_infer_and_save_txt(model_path, images_dir, output_txt_dir, output_vis_dir,
                           initial_conf=INITIAL_CONF, min_conf=MIN_CONF, conf_step=CONF_STEP,
                           traffic_class_name_candidates=("traffic sign","traffic_sign","traffic-sign")):
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    traffic_idx = read_sd_traffic_sign_index(SD_YAML, traffic_class_name_candidates)
    if traffic_idx is None:
        print("Info: traffic sign class not found; no filtering applied.")
    else:
        print(f"Traffic sign index: {traffic_idx}")

    conf = initial_conf
    filtered_txt_moved = []
    images_copied = 0

    while conf >= min_conf and len(filtered_txt_moved) == 0:
        print(f"\n---- YOLO predict (conf={conf}) ----")
        cmd = [
            "yolo", "detect", "predict",
            f"model={model_path}",
            f"source={images_dir}",
            "save=True",
            f"conf={conf}",
            f"project={PROJECT}",
            "name=pseudo_infer",
            "save_txt=True"
        ]
        subprocess.run(cmd, check=True)

        candidates = find_latest_pseudo_dirs(Path(PROJECT), name_substr="pseudo_infer")
        if not candidates:
            conf -= conf_step
            continue
        runs_detect_dir = candidates[0]

        txt_files = collect_label_txts_from_dir(runs_detect_dir / "labels") \
            if (runs_detect_dir / "labels").exists() else collect_label_txts_from_dir(runs_detect_dir)
        filtered = []
        for txt in txt_files:
            try:
                lines = open(txt).readlines()
            except:
                continue
            if traffic_idx is None:
                if lines: filtered.append((txt, lines))
            else:
                new_lines = [ln for ln in lines if not ln.strip().startswith(str(traffic_idx) + " ")]
                if new_lines: filtered.append((txt, new_lines))

        if not filtered:
            shutil.rmtree(runs_detect_dir, ignore_errors=True)
            conf -= conf_step
            continue

        for src_path, lines in filtered:
            dst_path = Path(output_txt_dir) / src_path.name
            with open(dst_path, "w") as f: f.writelines(lines)
        for ext in ("*.jpg","*.png"):
            for img in runs_detect_dir.rglob(ext):
                shutil.copy(img, Path(output_vis_dir) / img.name)

        filtered_txt_moved = list(Path(output_txt_dir).glob("*.txt"))

    print(f"{len(filtered_txt_moved)} pseudo labels ready.")
    return len(filtered_txt_moved), images_copied

# ---------------- Dataset merge ----------------
def prepare_merged_dataset(sd_images, sd_labels, td_images, td_pseudo_labels, out_dir):
    img_out = Path(out_dir) / "images" / "train"
    lbl_out = Path(out_dir) / "labels" / "train"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for f in Path(sd_images).glob("*.jpg"): shutil.copy(f, img_out / f.name)
    for f in Path(sd_labels).glob("*.txt"): shutil.copy(f, lbl_out / f.name)
    for f in Path(td_images).glob("*.jpg"): shutil.copy(f, img_out / f.name)
    for f in Path(td_pseudo_labels).glob("*.txt"): shutil.copy(f, lbl_out / f.name)
    print("Merged dataset prepared at:", out_dir)

def create_merged_yaml(sd_yaml, merged_dataset_dir, merged_yaml_path):
    with open(sd_yaml, "r") as f: data = yaml.safe_load(f)
    merged_dict = {
        "train": str(Path(merged_dataset_dir) / "images" / "train"),
        "val": data["val"],
        "test": data.get("test", data["val"]),
        "nc": data["nc"],
        "names": data["names"],
    }
    with open(merged_yaml_path, "w") as f: yaml.dump(merged_dict, f)
    print("Merged YAML created at:", merged_yaml_path)

def merge_and_train(merged_yaml, model_out_name="uda_st_train"):
    cmd = [
        "yolo", "detect", "train",
        f"model=yolov8l.pt",
        f"data={merged_yaml}",
        "epochs=60",
        f"project={PROJECT}",
        f"name={model_out_name}"
    ]
    subprocess.run(cmd, check=True)

# ---------------- Evaluation ----------------
def evaluate_model(model_path, yaml_path, project_dir=PROJECT):
    project_dir = Path(project_dir)

    # Validation
    subprocess.run(["yolo", "detect", "val", f"model={model_path}", f"data={yaml_path}", "plots=True"], check=True)
    # Test
    subprocess.run(["yolo", "detect", "val", f"model={model_path}", f"data={yaml_path}", "split=test", "plots=True"], check=True)

    # 결과 CSV 불러오기
    for split in ["val", "val"]:  # test도 동일 폴더에 저장됨
        results_csv = Path(project_dir) / "runs" / "detect" / split / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                last_row = list(csv.DictReader(f))[-1]
            print(f"\n==== {split.upper()} Metrics ====")
            print({k: last_row[k] for k in last_row.keys() if "metrics" in k or "mAP" in k})

    # 시각화 prediction
    with open(yaml_path) as f: data = yaml.safe_load(f)
    test_images_dir = Path(data.get("test", data.get("val")))
    vis_output_dir = project_dir / "runs" / "detect" / "test_vis"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "yolo", "detect", "predict",
        f"model={model_path}",
        f"source={test_images_dir}",
        "save=True", "save_txt=True",
        f"project={project_dir}", "name=test_vis"
    ], check=True)

    print("\n Evaluation Complete")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    run_infer_and_save_txt(SO_MODEL, TD_IMAGES_DIR, TD_PSEUDO_LABEL_DIR, TD_PSEUDO_VIS_DIR)
    prepare_merged_dataset(SD_IMAGES_DIR, SD_LABELS_DIR, TD_IMAGES_DIR, TD_PSEUDO_LABEL_DIR, MERGED_DATASET_DIR)
    create_merged_yaml(SD_YAML, MERGED_DATASET_DIR, MERGED_YAML)
    merge_and_train(MERGED_YAML, model_out_name="uda_st_train")
    evaluate_model(f"{PROJECT}/uda_st_train/weights/best.pt", MERGED_YAML)
