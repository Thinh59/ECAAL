"""
patch_notebook_v2.py  —  chạy bằng: python -X utf8 patch_notebook_v2.py
Thêm vào cv-eval-test.ipynb:
  - Đổi cell không cần chạy lại thành cell_type="raw" (Kaggle bỏ qua khi Save Version)
  - Thêm cell phân tích sâu: class imbalance, confusion-style analysis,
    feature-space t-SNE, training curve, overfitting table
"""
import json, copy
from pathlib import Path

NB  = Path("notebooks/cv-eval-test.ipynb")
nb  = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# ─── Cells KHÔNG cần chạy lại khi save version ───────────────────────────
# Đánh dấu bằng "raw" thay vì "code" — Kaggle Skip khi run all
SKIP_SUBSTRINGS = [
    "Cell 13: Ablation Study Results",   # md header
    "Cell 14: Plot Training Curves",     # md header
    "Cell 18: Cross-Dataset Comparison", # md header
    "## Cell 20: Sample Prediction",     # visualization md
]

def should_freeze(cell):
    src = "".join(cell.get("source", []))
    return any(s in src for s in SKIP_SUBSTRINGS)

# Freeze the code cells that follow frozen-md headers
freeze_next_code = False
new_cells = []
for cell in cells:
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and should_freeze(cell):
        freeze_next_code = True
        new_cells.append(cell)
        continue
    if freeze_next_code and cell.get("cell_type") == "code":
        cell = copy.deepcopy(cell)
        cell["cell_type"] = "raw"   # Kaggle ignores raw cells in batch run
        cell["metadata"]["tags"] = ["skip"]
        freeze_next_code = False
        new_cells.append(cell)
        continue
    freeze_next_code = False
    new_cells.append(cell)

# ─── Deep Analysis Cell source ────────────────────────────────────────────
ANALYSIS_SOURCE = [
"# ============================================================",
"# DEEP ANALYSIS CELL — Class Imbalance + Overfitting + Curves",
"# ============================================================",
"import json, os",
"import numpy as np",
"import pandas as pd",
"import matplotlib.pyplot as plt",
"import matplotlib.gridspec as gridspec",
"from pathlib import Path",
"",
"OUTPUTS_DIR = Path('/kaggle/working/outputs')",
"",
"# ── 1. Load all log.json ──────────────────────────────────────",
"EXPS = {",
"    'A: ResNet50+BCE':           'exp_A_resnet_bce',",
"    'B: ResNet50+ASL':           'exp_B_resnet_asl',",
"    'C: EffNet+CBAM+ASL':        'exp_C_efficientnet_cbam_asl',",
"    'D: ResNet50+Focal':         'exp_D_resnet_focal',",
"    'E: ResNet50+CBAM+ASL':      'exp_E_resnet_cbam_asl',",
"    'F: EffNet+ASL':             'exp_F_efficientnet_asl',",
"    'G: EffNet+CBAM+ASL+AThres': 'exp_G_efficientnet_cbam_asl',",
"}",
"",
"logs = {}",
"for name, d in EXPS.items():",
"    p = OUTPUTS_DIR / d / 'log.json'",
"    if p.exists():",
"        logs[name] = json.loads(p.read_text())",
"",
"# ── 2. Overfitting Analysis Table ─────────────────────────────",
"ov_csv = OUTPUTS_DIR / 'overfitting_analysis.csv'",
"if ov_csv.exists():",
"    df_ov = pd.read_csv(ov_csv)",
"    print('\\n' + '='*65)",
"    print('OVERFITTING ANALYSIS')",
"    print('='*65)",
"    print(df_ov.to_string(index=False))",
"",
"# ── 3. Training Curves: mAP + loss per epoch ─────────────────",
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
"fig.suptitle('Training Curves — All Experiments', fontsize=13, fontweight='bold')",
"",
"colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))",
"for ax in axes: ax.grid(True, alpha=0.3)",
"",
"for (name, records), color in zip(logs.items(), colors):",
"    epochs  = [r['epoch']      for r in records]",
"    maps    = [r['mAP']        for r in records]",
"    losses  = [r.get('train_loss', float('nan')) for r in records]",
"    axes[0].plot(epochs, maps,   label=name, color=color, linewidth=1.8)",
"    axes[1].plot(epochs, losses, label=name, color=color, linewidth=1.8)",
"",
"axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val mAP')",
"axes[0].set_title('Val mAP over Epochs'); axes[0].legend(fontsize=7)",
"axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Train Loss')",
"axes[1].set_title('Train Loss over Epochs'); axes[1].legend(fontsize=7)",
"",
"plt.tight_layout()",
"plt.savefig(str(OUTPUTS_DIR / 'training_curves_all.png'), dpi=120)",
"plt.show()",
"print('Saved: training_curves_all.png')",
"",
"# ── 4. Class Imbalance Bar Chart ─────────────────────────────",
"# Count positive labels per class from test set",
"import torch, sys",
"sys.path.insert(0, '/kaggle/working/ECAAL/src')",
"from dataset import COCOMultiLabelDataset, get_val_transform",
"",
"COCO_ROOT  = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'",
"SUBSET_DIR = Path('/kaggle/working/data/coco_subset')",
"import json as _json",
"",
"COCO_CATS = [",
"    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',",
"    'traffic light','fire hydrant','stop sign','parking meter','bench','bird',",
"    'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',",
"    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',",
"    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',",
"    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',",
"    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',",
"    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',",
"    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',",
"    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',",
"    'hair drier','toothbrush'",
"]",
"",
"transform  = get_val_transform(img_size=224)",
"train_ids  = _json.load(open(SUBSET_DIR / 'subset_train_ids.json'))",
"train_ds   = COCOMultiLabelDataset(COCO_ROOT, 'train', transform, train_ids)",
"",
"# Accumulate label counts",
"label_counts = np.zeros(80, dtype=int)",
"for _, label in train_ds:",
"    label_counts += label.numpy().astype(int)",
"",
"# Sort and plot",
"order   = np.argsort(label_counts)[::-1]",
"fig, ax = plt.subplots(figsize=(18, 4))",
"bars = ax.bar(range(80), label_counts[order],",
"              color=plt.cm.RdYlGn(label_counts[order] / label_counts.max()))",
"ax.set_xticks(range(80))",
"ax.set_xticklabels([COCO_CATS[i] for i in order], rotation=90, fontsize=7)",
"ax.set_ylabel('# positive samples in train set')",
"ax.set_title('Class Imbalance — MS COCO Subset (16k train)', fontweight='bold')",
"ax.axhline(label_counts.mean(), color='red', linestyle='--', label=f'Mean={label_counts.mean():.0f}')",
"ax.legend()",
"plt.tight_layout()",
"plt.savefig(str(OUTPUTS_DIR / 'class_imbalance.png'), dpi=120)",
"plt.show()",
"print(f'Most frequent: {COCO_CATS[order[0]]}={label_counts[order[0]]:,}')",
"print(f'Least frequent: {COCO_CATS[order[-1]]}={label_counts[order[-1]]:,}')",
"print(f'Imbalance ratio: {label_counts.max() / label_counts.min():.1f}x')",
"",
"# ── 5. Per-class AP comparison: C vs G ───────────────────────",
"# Load best epoch AP_per_class for Exp C and G",
"def best_ap(name):",
"    d = EXPS.get(name)",
"    if not d: return None",
"    p = OUTPUTS_DIR / d / 'log.json'",
"    if not p.exists(): return None",
"    records = json.loads(p.read_text())",
"    best = max(records, key=lambda r: r.get('mAP', 0))",
"    return np.array(best.get('AP_per_class', []))",
"",
"ap_c = best_ap('C: EffNet+CBAM+ASL')",
"ap_g = best_ap('G: EffNet+CBAM+ASL+AThres')",
"",
"if ap_c is not None and ap_g is not None:",
"    fig, ax = plt.subplots(figsize=(18, 4))",
"    x = np.arange(80)",
"    ax.bar(x - 0.2, ap_c, 0.4, label='Exp C (no AThres)', alpha=0.8, color='#e05c5c')",
"    ax.bar(x + 0.2, ap_g, 0.4, label='Exp G (AThres)',    alpha=0.8, color='#5ca8e0')",
"    ax.set_xticks(x)",
"    ax.set_xticklabels(COCO_CATS, rotation=90, fontsize=6)",
"    ax.set_ylabel('AP')",
"    ax.set_title('Per-class AP: Exp C vs Exp G (Val set)', fontweight='bold')",
"    ax.legend()",
"    plt.tight_layout()",
"    plt.savefig(str(OUTPUTS_DIR / 'perclass_ap_C_vs_G.png'), dpi=120)",
"    plt.show()",
"    delta = ap_g - ap_c",
"    improved = np.sum(delta > 0)",
"    print(f'Classes improved in G vs C: {improved}/80')",
"    print(f'Top 5 improved: {[(COCO_CATS[i], f\"+{delta[i]:.3f}\") for i in np.argsort(delta)[::-1][:5]]}')",
]

def make_code_cell(cell_id, source_lines):
    lines = []
    for i, line in enumerate(source_lines):
        lines.append(line + ("\n" if i < len(source_lines) - 1 else ""))
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": []},
        "outputs": [],
        "source": lines,
    }

def make_md_cell(cell_id, text):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": []},
        "source": [text],
    }

# ─── Insert analysis cells before the Zip cell ───────────────────────────
final_cells = []
for cell in new_cells:
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "Cell 21: Zip" in src:
        final_cells.append(make_md_cell("analysis_md", "## Deep Analysis: Class Imbalance, Overfitting & Training Curves\n"))
        final_cells.append(make_code_cell("analysis_code", ANALYSIS_SOURCE))
        print("[PATCH v2] Inserted deep analysis cell")
    final_cells.append(cell)

nb["cells"] = final_cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Done. Total cells: {len(final_cells)}")
