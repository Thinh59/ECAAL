"""
patch_notebook.py
-----------------
Patches cv-eval-test.ipynb to:
  1. Add Exp G to Cell 17 (COCO Test evaluation)
  2. Add Exp G to Cell 19 (Overfitting Analysis)
  3. Replace Cell 20 (Zip) with a new Visualization cell + keep Zip as Cell 21
Run from repo root:
    python patch_notebook.py
"""

import json
from pathlib import Path
import copy

NB_PATH = Path(__file__).parent / "notebooks" / "cv-eval-test.ipynb"

# ─── helpers ────────────────────────────────────────────────────────────────
def src(lines):
    """Return a 'source' list (each line already has \\n except last)."""
    result = []
    for i, line in enumerate(lines):
        result.append(line + ("\n" if i < len(lines) - 1 else ""))
    return result


def code_cell(cell_id, source_lines, execution_count=None, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "id": cell_id,
        "metadata": {"tags": []},
        "outputs": outputs or [],
        "source": src(source_lines),
    }


def md_cell(cell_id, text):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": []},
        "source": [text],
    }


# ─── Exp G row ───────────────────────────────────────────────────────────────
EXP_G_ROW = "    ('G: EffNet+CBAM+ASL+AThres', 'exp_G_efficientnet_cbam_asl', 'efficientnet_b0', True),"

# ─── Visualization cell source ───────────────────────────────────────────────
VIS_SOURCE = [
    "import torch, json",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "from pathlib import Path",
    "import sys",
    "",
    "sys.path.insert(0, str(REPO_DIR / 'src'))",
    "from models import build_model",
    "from dataset import COCOMultiLabelDataset, get_val_transform",
    "",
    "COCO_ROOT   = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'",
    "OUTPUTS_DIR = Path('/kaggle/working/outputs')",
    "SUBSET_DIR  = Path('/kaggle/working/data/coco_subset')",
    "DEVICE      = 'cuda'",
    "",
    "COCO_CATS = [",
    "    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',",
    "    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',",
    "    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',",
    "    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',",
    "    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',",
    "    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',",
    "    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',",
    "    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',",
    "    'scissors','teddy bear','hair drier','toothbrush'",
    "]",
    "",
    "EXPERIMENTS_VIS = [",
    "    ('C: EffNet+CBAM+ASL',        'exp_C_efficientnet_cbam_asl', 'efficientnet_b0', True),",
    "    ('G: EffNet+CBAM+ASL+AThres', 'exp_G_efficientnet_cbam_asl', 'efficientnet_b0', True),",
    "    ('B: ResNet50+ASL',           'exp_B_resnet_asl',            'resnet50',        False),",
    "]",
    "",
    "THRESHOLD = 0.5",
    "N_SAMPLES  = 6  # worst-case images per experiment",
    "",
    "transform = get_val_transform(img_size=224)",
    "test_ids  = json.load(open(SUBSET_DIR / 'subset_test_ids.json'))",
    "test_ds   = COCOMultiLabelDataset(COCO_ROOT, 'val', transform, test_ids)",
    "",
    "def denormalize(tensor):",
    "    mean = np.array([0.485, 0.456, 0.406])",
    "    std  = np.array([0.229, 0.224, 0.225])",
    "    img  = tensor.permute(1,2,0).numpy()",
    "    img  = img * std + mean",
    "    return np.clip(img, 0, 1)",
    "",
    "def visualize_samples(name, exp_dir, backbone, use_cbam):",
    "    pth = OUTPUTS_DIR / exp_dir / 'best.pth'",
    "    if not pth.exists():",
    "        print(f'Skip {name}: no best.pth'); return",
    "    model = build_model({'backbone': backbone, 'use_cbam': use_cbam,",
    "                         'num_classes': 80, 'pretrained': False}).to(DEVICE)",
    "    ckpt  = torch.load(pth, map_location=DEVICE, weights_only=False)",
    "    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)",
    "    model.eval()",
    "",
    "    errors = []",
    "    with torch.no_grad():",
    "        for idx in range(min(500, len(test_ds))):",
    "            img_t, target = test_ds[idx]",
    "            logit = model(img_t.unsqueeze(0).to(DEVICE))",
    "            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()",
    "            pred  = (prob >= THRESHOLD).astype(float)",
    "            gt    = target.numpy()",
    "            err   = int(np.sum(np.abs(pred - gt)))",
    "            errors.append((err, idx, img_t, gt, prob, pred))",
    "",
    "    errors.sort(key=lambda x: -x[0])",
    "    worst = errors[:N_SAMPLES]",
    "",
    "    fig, axes = plt.subplots(2, N_SAMPLES, figsize=(N_SAMPLES * 3.5, 7))",
    "    fig.suptitle(f'Sample Predictions — {name}  (worst {N_SAMPLES} by FP+FN)', fontsize=12, fontweight='bold')",
    "",
    "    for col, (err, idx, img_t, gt, prob, pred) in enumerate(worst):",
    "        img_np = denormalize(img_t)",
    "        gt_labels   = [COCO_CATS[i] for i in range(80) if gt[i]   > 0.5]",
    "        pred_labels = [COCO_CATS[i] for i in range(80) if pred[i] > 0.5]",
    "        tp = set(gt_labels) & set(pred_labels)",
    "        fp = set(pred_labels) - set(gt_labels)",
    "        fn = set(gt_labels)   - set(pred_labels)",
    "",
    "        axes[0, col].imshow(img_np)",
    "        axes[0, col].axis('off')",
    "        axes[0, col].set_title(f'Errors={err}', fontsize=9)",
    "",
    "        axes[1, col].axis('off')",
    "        txt = ''",
    "        if tp: txt += 'TP: ' + ', '.join(sorted(tp)) + '\\n'",
    "        if fp: txt += 'FP: ' + ', '.join(sorted(fp)) + '\\n'",
    "        if fn: txt += 'FN: ' + ', '.join(sorted(fn))",
    "        axes[1, col].text(0.5, 0.98, txt, ha='center', va='top', fontsize=7,",
    "                          transform=axes[1, col].transAxes,",
    "                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))",
    "",
    "    plt.tight_layout()",
    "    save_path = OUTPUTS_DIR / f'vis_samples_{exp_dir}.png'",
    "    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')",
    "    plt.show()",
    "    print(f'Saved: {save_path}')",
    "",
    "for name, exp_dir, backbone, use_cbam in EXPERIMENTS_VIS:",
    "    print(f'\\n=== Visualizing: {name} ===')",
    "    visualize_samples(name, exp_dir, backbone, use_cbam)",
]


def patch(nb):
    cells = nb["cells"]
    new_cells = []

    for cell in cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        src_text = "".join(cell.get("source", []))

        # ── Cell 17: COCO Test evaluation ───────────────────────────────────
        if ("coco_test_results = []" in src_text and
                "COCO TEST RESULTS" in src_text and
                EXP_G_ROW not in src_text):

            new_source = []
            exp_list_closed = False
            for line in cell["source"]:
                stripped = line.rstrip("\n")
                # Inject Exp G just before closing bracket of EXPERIMENTS
                if not exp_list_closed and stripped.strip() == "]":
                    # Check we are closing the EXPERIMENTS list (look back)
                    new_source.append("    " + EXP_G_ROW + "\n")
                    exp_list_closed = True
                new_source.append(line)
            cell = copy.deepcopy(cell)
            cell["source"] = new_source
            print("[PATCH] Cell 17 (COCO Test): injected Exp G row")

        # ── Cell 19: Overfitting Analysis ────────────────────────────────────
        elif ("OVERFITTING ANALYSIS" in src_text and
              "train_loader" in src_text and
              EXP_G_ROW not in src_text):

            new_source = []
            exp_list_closed = False
            for line in cell["source"]:
                stripped = line.rstrip("\n")
                if not exp_list_closed and stripped.strip() == "]":
                    new_source.append("    " + EXP_G_ROW + "\n")
                    exp_list_closed = True
                new_source.append(line)
            cell = copy.deepcopy(cell)
            cell["source"] = new_source
            print("[PATCH] Cell 19 (Overfitting): injected Exp G row")

        new_cells.append(cell)

    # ── Insert visualization cell + update Cell 20 heading ──────────────────
    final_cells = []
    for i, cell in enumerate(new_cells):
        src_text = "".join(cell.get("source", []))
        # Detect the old "Cell 20: Zip" markdown header
        if cell.get("cell_type") == "markdown" and "Cell 20: Zip" in src_text:
            # Insert visualization md + code cells BEFORE the zip header
            final_cells.append(md_cell("cell_vis_md", "## Cell 20: Sample Prediction Visualization (worst FP+FN)\n"))
            final_cells.append(code_cell("cell_vis_code", VIS_SOURCE))
            # Change the zip header to Cell 21
            cell = copy.deepcopy(cell)
            cell["source"] = ["## Cell 21: Zip & Download Results\n"]
            print("[PATCH] Inserted visualization cell before Zip cell")

        final_cells.append(cell)

    nb["cells"] = final_cells
    return nb


def main():
    print(f"Loading {NB_PATH}")
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb = patch(nb)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Done — saved {NB_PATH}")


if __name__ == "__main__":
    main()
