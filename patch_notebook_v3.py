"""
patch_notebook_v3.py  — python -X utf8 patch_notebook_v3.py
Thêm 3 thứ còn thiếu vào cv-eval-test.ipynb:
  1. Unfreeze Cell 20 (sample visualization): raw -> code
  2. Thêm cell Co-occurrence heatmap (thay confusion matrix)
  3. Thêm cell t-SNE feature-space visualization
"""
import json, copy
from pathlib import Path

NB = Path("notebooks/cv-eval-test.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


# ── helpers ────────────────────────────────────────────────────────────────
def src_lines(lines):
    out = []
    for i, l in enumerate(lines):
        out.append(l + ("\n" if i < len(lines) - 1 else ""))
    return out


def code_cell(cid, lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {"tags": []},
        "outputs": [],
        "source": src_lines(lines),
    }


def md_cell(cid, text):
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {"tags": []},
        "source": [text],
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. Co-occurrence heatmap source
# ══════════════════════════════════════════════════════════════════════════
COOC_SOURCE = [
    "# Co-occurrence Heatmap — which classes appear together?",
    "# (Multi-label substitute for confusion matrix)",
    "import numpy as np, json, sys",
    "import matplotlib.pyplot as plt",
    "import matplotlib.colors as mcolors",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(REPO_DIR / 'src'))",
    "from dataset import COCOMultiLabelDataset, get_val_transform",
    "",
    "COCO_ROOT  = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'",
    "SUBSET_DIR = Path('/kaggle/working/data/coco_subset')",
    "OUTPUTS_DIR= Path('/kaggle/working/outputs')",
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
    "# ── Load train labels ──────────────────────────────────────",
    "transform = get_val_transform(img_size=224)",
    "train_ids = json.load(open(SUBSET_DIR / 'subset_train_ids.json'))",
    "train_ds  = COCOMultiLabelDataset(COCO_ROOT, 'train', transform, train_ids)",
    "",
    "labels = []",
    "for _, lbl in train_ds:",
    "    labels.append(lbl.numpy())",
    "L = np.array(labels)   # (N, 80)  binary",
    "N = len(L)",
    "",
    "# ── Co-occurrence matrix (normalised by sqrt of marginals) ─",
    "# co_ij = P(i AND j) / sqrt(P(i)*P(j))  — like Phi coefficient",
    "freq  = L.sum(0) / N                           # (80,)",
    "cooc  = (L.T @ L) / N                          # (80,80)",
    "denom = np.sqrt(np.outer(freq, freq))",
    "denom[denom == 0] = 1",
    "phi   = cooc / denom                           # normalised co-occurrence",
    "np.fill_diagonal(phi, 0)                        # hide self-co-occurrence",
    "",
    "# ── Pick top-30 classes by frequency for readability ──────",
    "top30 = np.argsort(freq)[::-1][:30]",
    "phi30 = phi[np.ix_(top30, top30)]",
    "cats30 = [COCO_CATS[i] for i in top30]",
    "",
    "fig, ax = plt.subplots(figsize=(14, 12))",
    "im = ax.imshow(phi30, cmap='YlOrRd', aspect='auto', vmin=0, vmax=phi30.max())",
    "ax.set_xticks(range(30)); ax.set_xticklabels(cats30, rotation=90, fontsize=8)",
    "ax.set_yticks(range(30)); ax.set_yticklabels(cats30, fontsize=8)",
    "plt.colorbar(im, ax=ax, fraction=0.03, label='Phi coefficient')",
    "ax.set_title('Co-occurrence Heatmap (Top-30 frequent classes, Train set)',",
    "             fontsize=12, fontweight='bold')",
    "",
    "# Annotate top co-occurring pairs",
    "for i in range(30):",
    "    for j in range(30):",
    "        if phi30[i,j] > 0.4:",
    "            ax.text(j, i, f'{phi30[i,j]:.2f}', ha='center', va='center',",
    "                    fontsize=6, color='black')",
    "",
    "plt.tight_layout()",
    "plt.savefig(str(OUTPUTS_DIR / 'cooccurrence_heatmap.png'), dpi=120)",
    "plt.show()",
    "",
    "# ── Print top-10 confused pairs ───────────────────────────",
    "pairs = []",
    "for i in range(80):",
    "    for j in range(i+1, 80):",
    "        pairs.append((phi[i,j], COCO_CATS[i], COCO_CATS[j]))",
    "pairs.sort(reverse=True)",
    "print('Top-10 co-occurring class pairs (Phi coefficient):')",
    "for v, a, b in pairs[:10]:",
    "    print(f'  {a:20s} <-> {b:20s}  phi={v:.3f}')",
    "",
    "# ── FP analysis: which classes cause FP for each other? ───",
    "# Use Exp C predictions on test set",
    "import torch",
    "from models import build_model",
    "",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "test_ids = json.load(open(SUBSET_DIR / 'subset_test_ids.json'))",
    "test_ds  = COCOMultiLabelDataset(COCO_ROOT, 'val', transform, test_ids)",
    "test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64,",
    "                                           num_workers=2, pin_memory=True)",
    "",
    "pth = OUTPUTS_DIR / 'exp_C_efficientnet_cbam_asl' / 'best.pth'",
    "if pth.exists():",
    "    model = build_model({'backbone':'efficientnet_b0','use_cbam':True,",
    "                         'num_classes':80,'pretrained':False}).to(DEVICE)",
    "    ckpt = torch.load(pth, map_location=DEVICE, weights_only=False)",
    "    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)",
    "    model.eval()",
    "    all_p, all_t = [], []",
    "    with torch.no_grad():",
    "        for imgs, tgts in test_loader:",
    "            p = torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy()",
    "            all_p.append(p); all_t.append(tgts.numpy())",
    "    P = np.concatenate(all_p); T = np.concatenate(all_t)",
    "    pred = (P >= 0.5).astype(int)",
    "    # FP matrix: FP[i,j] = times class i is False-Positive when class j is GT=1",
    "    FP_mat = np.zeros((80, 80), dtype=int)",
    "    for n in range(len(T)):",
    "        fp_cls = np.where((pred[n]==1) & (T[n]==0))[0]",
    "        gt_cls = np.where(T[n]==1)[0]",
    "        for fp in fp_cls:",
    "            for gt in gt_cls:",
    "                FP_mat[fp, gt] += 1",
    "    # Top FP pairs",
    "    fp_pairs = []",
    "    for i in range(80):",
    "        for j in range(80):",
    "            if i != j and FP_mat[i,j] > 0:",
    "                fp_pairs.append((FP_mat[i,j], COCO_CATS[i], COCO_CATS[j]))",
    "    fp_pairs.sort(reverse=True)",
    "    print('\\nTop-10 FP confusion pairs (class A predicted when class B is GT):')",
    "    for cnt, a, b in fp_pairs[:10]:",
    "        print(f'  Predict [{a:20s}]  when GT=[{b:20s}]  count={cnt}')",
    "else:",
    "    print('Exp C best.pth not found — skip FP analysis')",
]

# ══════════════════════════════════════════════════════════════════════════
# 2. t-SNE feature-space visualization source
# ══════════════════════════════════════════════════════════════════════════
TSNE_SOURCE = [
    "# t-SNE Feature Space Visualization",
    "# Extract 1280-dim GAP vectors, project to 2D, colour by most-frequent label",
    "import torch, json, sys",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "from pathlib import Path",
    "from sklearn.manifold import TSNE",
    "from sklearn.decomposition import PCA",
    "",
    "sys.path.insert(0, str(REPO_DIR / 'src'))",
    "from models import build_model",
    "from dataset import COCOMultiLabelDataset, get_val_transform",
    "",
    "COCO_ROOT  = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'",
    "SUBSET_DIR = Path('/kaggle/working/data/coco_subset')",
    "OUTPUTS_DIR= Path('/kaggle/working/outputs')",
    "DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'",
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
    "# ── Hook to capture GAP output ────────────────────────────",
    "features = {}",
    "def hook_fn(m, inp, out):",
    "    features['gap'] = out.detach().cpu()   # (B, 1280)",
    "",
    "pth = OUTPUTS_DIR / 'exp_C_efficientnet_cbam_asl' / 'best.pth'",
    "if not pth.exists():",
    "    print('Exp C best.pth not found'); raise SystemExit",
    "",
    "model = build_model({'backbone':'efficientnet_b0','use_cbam':True,",
    "                     'num_classes':80,'pretrained':False}).to(DEVICE)",
    "ckpt  = torch.load(pth, map_location=DEVICE, weights_only=False)",
    "model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)",
    "model.eval()",
    "",
    "# Register hook on the adaptive_avg_pool2d (GAP) layer",
    "# EfficientNet: model.backbone.avgpool or equivalent",
    "hook = model.gap.register_forward_hook(hook_fn)",
    "",
    "# ── Extract features from val set (1000 samples) ──────────",
    "transform = get_val_transform(img_size=224)",
    "val_ids   = json.load(open(SUBSET_DIR / 'subset_val_ids.json'))",
    "val_ds    = COCOMultiLabelDataset(COCO_ROOT, 'val', transform, val_ids)",
    "val_loader= torch.utils.data.DataLoader(val_ds, batch_size=64,",
    "                                         num_workers=2, pin_memory=True)",
    "",
    "all_feats, all_labels = [], []",
    "with torch.no_grad():",
    "    for imgs, lbls in val_loader:",
    "        _ = model(imgs.to(DEVICE))",
    "        all_feats.append(features['gap'].squeeze(-1).squeeze(-1).numpy()",
    "                         if features['gap'].ndim == 4",
    "                         else features['gap'].numpy())",
    "        all_labels.append(lbls.numpy())",
    "",
    "hook.remove()",
    "F = np.concatenate(all_feats,  axis=0)   # (N, 1280)",
    "L = np.concatenate(all_labels, axis=0)   # (N, 80)",
    "print(f'Feature matrix: {F.shape}')",
    "",
    "# ── PCA 50 -> t-SNE 2 ─────────────────────────────────────",
    "print('Running PCA (1280->50)...')",
    "pca  = PCA(n_components=50, random_state=42)",
    "F50  = pca.fit_transform(F)",
    "print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')",
    "",
    "print('Running t-SNE (50->2)... (~2 min)')",
    "tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, verbose=1)",
    "F2   = tsne.fit_transform(F50)",
    "",
    "# ── Plot: colour by single most-confident label ────────────",
    "# Choose top-8 frequent classes to colour",
    "freq_order = np.argsort(L.sum(0))[::-1][:8]",
    "label_colors = plt.cm.Set1(np.linspace(0, 0.9, 8))",
    "",
    "fig, ax = plt.subplots(figsize=(13, 10))",
    "# background (multi-label, uncoloured)",
    "ax.scatter(F2[:,0], F2[:,1], c='lightgray', s=8, alpha=0.3, label='other')",
    "",
    "for k, cls_idx in enumerate(freq_order):",
    "    mask = L[:, cls_idx] == 1",
    "    ax.scatter(F2[mask, 0], F2[mask, 1],",
    "               c=[label_colors[k]], s=12, alpha=0.7,",
    "               label=COCO_CATS[cls_idx])",
    "",
    "ax.set_title('t-SNE of GAP Feature Vectors — Exp C (Val set, 1k samples)',",
    "             fontsize=12, fontweight='bold')",
    "ax.legend(markerscale=2, fontsize=9, loc='best')",
    "ax.axis('off')",
    "plt.tight_layout()",
    "plt.savefig(str(OUTPUTS_DIR / 'tsne_features_expC.png'), dpi=120)",
    "plt.show()",
    "print('Saved: tsne_features_expC.png')",
    "",
    "# ── Quantify overlap: avg pairwise cosine distance per class ─",
    "from sklearn.metrics.pairwise import cosine_distances",
    "print('\\nIntra-class vs Inter-class distance (cosine, top-5 classes):')",
    "print(f'  {\"Class\":20s}  {\"Intra\":>8}  {\"Inter\":>8}')",
    "cls_feats = {cls_idx: F[L[:,cls_idx]==1] for cls_idx in freq_order if L[:,cls_idx].sum()>1}",
    "for cls_idx, cf in list(cls_feats.items())[:5]:",
    "    intra = cosine_distances(cf).mean()",
    "    others = F[L[:,cls_idx]==0]",
    "    inter  = cosine_distances(cf, others[:200]).mean()",
    "    print(f'  {COCO_CATS[cls_idx]:20s}  {intra:8.4f}  {inter:8.4f}')",
    "print('  -> Small intra + large inter = clear cluster (easy class)')",
    "print('  -> Large intra or small inter = confused class (hard)')",
]

# ══════════════════════════════════════════════════════════════════════════
# Apply patches
# ══════════════════════════════════════════════════════════════════════════
cells = nb["cells"]
new_cells = []

for cell in cells:
    # ── Fix 1: Unfreeze Cell 20 (sample vis) ──────────────────
    if (cell.get("cell_type") == "raw" and
            cell.get("id") == "cell_vis_code"):
        cell = copy.deepcopy(cell)
        cell["cell_type"] = "code"
        cell["metadata"]["tags"] = []
        print("[PATCH v3] Unfreeze cell_vis_code (Cell 20)")

    new_cells.append(cell)

# ── Fix 2 & 3: Insert Co-occ + t-SNE before Deep Analysis md ──
final_cells = []
for cell in new_cells:
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "Deep Analysis:" in src:
        # Insert co-occurrence + t-SNE BEFORE existing deep analysis
        final_cells.append(md_cell("cooc_md",
            "## Co-occurrence Heatmap (Multi-label Confusion Matrix)\n"))
        final_cells.append(code_cell("cooc_code", COOC_SOURCE))
        final_cells.append(md_cell("tsne_md",
            "## t-SNE Feature Space Visualization (GAP vectors)\n"))
        final_cells.append(code_cell("tsne_code", TSNE_SOURCE))
        print("[PATCH v3] Inserted Co-occurrence + t-SNE cells")
    final_cells.append(cell)

nb["cells"] = final_cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Done. Total cells: {len(final_cells)}")
