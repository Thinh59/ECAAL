import json
from pathlib import Path

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Phân Tích Chuyên Sâu COCO vs Pascal VOC 2012\n",
        "Notebook này tập trung giải quyết các vấn đề:\n",
        "1. Điều tra tại sao mAP trên VOC lại bằng 0.0.\n",
        "2. Tìm các lớp dễ bị nhầm lẫn nhất từ Confusion Matrix trên COCO và trực quan hoá các feature của chúng xem có overlap không.\n",
        "3. Trực quan hoá sự khác biệt phân phối đặc trưng (Domain Shift) giữa tập dữ liệu COCO và Pascal VOC."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install timm pycocotools torchmetrics scikit-learn matplotlib seaborn pyyaml tqdm umap-learn -q\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import os\n",
        "from pathlib import Path\n",
        "print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Chuẩn bị Môi trường và Dữ liệu"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Clone repository\n",
        "REPO_URL = 'https://github.com/Thinh59/ECAAL.git'\n",
        "REPO_DIR = Path('/kaggle/working/ECAAL')\n",
        "if REPO_DIR.exists():\n",
        "    os.system(f'git -C {REPO_DIR} pull')\n",
        "else:\n",
        "    os.system(f'git clone {REPO_URL} {REPO_DIR}')\n",
        "import sys\n",
        "sys.path.insert(0, str(REPO_DIR / 'src'))"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import shutil\n",
        "OUTPUTS_DIR = Path('/kaggle/working/outputs')\n",
        "OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "DATASET_ROOT = Path('/kaggle/input/datasets/thinhha59/models')\n",
        "RESULTS_DIR  = DATASET_ROOT / 'results'\n",
        "SRC_SUBSET   = RESULTS_DIR / 'data' / 'coco_subset'\n",
        "SUBSET_DIR = Path('/kaggle/working/data/coco_subset')\n",
        "SUBSET_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "if RESULTS_DIR.exists():\n",
        "    for fname in ['subset_train_ids.json', 'subset_val_ids.json', 'subset_test_ids.json']:\n",
        "        if (SRC_SUBSET / fname).exists():\n",
        "            shutil.copy2(SRC_SUBSET / fname, SUBSET_DIR / fname)\n",
        "    exp = 'exp_C_efficientnet_cbam_asl'\n",
        "    src = RESULTS_DIR / exp\n",
        "    dst = OUTPUTS_DIR / exp\n",
        "    dst.mkdir(parents=True, exist_ok=True)\n",
        "    for fname in ['best.pth', 'log.json']:\n",
        "        if (src / fname).exists(): shutil.copy2(src / fname, dst / fname)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "coco_root = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'\n",
        "voc_root = '/kaggle/input/pascal-voc-2012/VOC2012'\n",
        "if os.path.exists(coco_root): print('COCO 2017 dataset found')\n",
        "if os.path.exists(voc_root): print('VOC 2012 dataset found')\n",
        "else: print('Vui lòng Add Data -> pascal-voc-2012 (kích thước khoảng ~2GB)')"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from dataset import COCOMultiLabelDataset, VOCMultiLabelDataset, get_val_transform\n",
        "from torch.utils.data import DataLoader\n",
        "import json\n",
        "\n",
        "transform = get_val_transform(224)\n",
        "\n",
        "# Load COCO Test\n",
        "test_ids = []\n",
        "test_ids_file = SUBSET_DIR / 'subset_test_ids.json'\n",
        "if test_ids_file.exists():\n",
        "    test_ids = json.load(open(test_ids_file))\n",
        "coco_test = COCOMultiLabelDataset(coco_root, split='val', transform=transform, subset_ids=test_ids)\n",
        "coco_loader = DataLoader(coco_test, batch_size=32, num_workers=2, shuffle=False)\n",
        "\n",
        "# Load VOC Val\n",
        "voc_val = VOCMultiLabelDataset(voc_root, split='val', transform=transform)\n",
        "voc_loader = DataLoader(voc_val, batch_size=32, num_workers=2, shuffle=False)"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Điều tra lỗi mAP VOC = 0.0\n",
        "Kiểm tra nhãn Ground Truth của VOC xem có bằng 0 hết không? \n",
        "Kiểm tra class mapping giữa COCO và VOC xem có đúng không?"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"Tổng số ảnh trong VOC val:\", len(voc_val))\n",
        "positive_labels = 0\n",
        "for img, label in voc_val:\n",
        "    positive_labels += label.sum().item()\n",
        "print(\"Tổng số nhãn dương trong VOC val:\", positive_labels)\n",
        "if positive_labels == 0:\n",
        "    print(\"!!! CẢNH BÁO: Tập VOC không chứa bất kỳ nhãn dương nào. Đây là nguyên nhân khiến mAP = 0.0!\")\n",
        "    print(\"Lý do: File annotation của VOC (.txt) có thể chứa khoảng trắng hoặc format khác với code parser.\")\n",
        "else:\n",
        "    print(\"Labels của VOC bình thường. Nguyên nhân mAP = 0.0 có thể do COCO_CLASSES không khớp.\")\n",
        "\n",
        "# Kiểm tra mapping\n",
        "from cross_evaluate import get_coco_to_voc_mapping, VOC_CLASSES\n",
        "mapping = get_coco_to_voc_mapping()\n",
        "print(\"\\nMapping từ VOC index sang COCO index:\")\n",
        "for voc_idx, coco_idx in mapping.items():\n",
        "    print(f\"VOC: {VOC_CLASSES[voc_idx]} -> COCO: ID {coco_idx}\")\n",
        "if len(mapping) < 20:\n",
        "    print(f\"\\n!!! CẢNH BÁO: Chỉ ánh xạ được {len(mapping)}/20 lớp. Các lớp bị thiếu sẽ khiến mAP giảm mạnh!\")\n"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Trích xuất đặc trưng (Feature Extraction) cho COCO và VOC"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from models import build_model\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "model_C = build_model({'backbone': 'efficientnet_b0', 'use_cbam': True, 'num_classes': 80}).to(device)\n",
        "ckpt_path = OUTPUTS_DIR / 'exp_C_efficientnet_cbam_asl' / 'best.pth'\n",
        "if ckpt_path.exists():\n",
        "    model_C.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])\n",
        "model_C.eval()\n",
        "\n",
        "# Dùng Forward Hook để lấy đặc trưng từ lớp Global Average Pooling (GAP)\n",
        "features_dict = {}\n",
        "def hook_fn(m, i, o):\n",
        "    features_dict['feat'] = o.detach().cpu().flatten(1)\n",
        "model_C.gap.register_forward_hook(hook_fn)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from tqdm.auto import tqdm\n",
        "def extract_features(loader):\n",
        "    all_feats, all_targets, all_preds = [], [], []\n",
        "    with torch.no_grad():\n",
        "        for imgs, targets in tqdm(loader, leave=False):\n",
        "            if isinstance(targets, tuple): # Dataloader trả về tuple nếu có thêm image_id\n",
        "                targets = targets[0] \n",
        "            out = torch.sigmoid(model_C(imgs.to(device)))\n",
        "            all_preds.append(out.cpu())\n",
        "            all_feats.append(features_dict['feat'])\n",
        "            all_targets.append(targets)\n",
        "    return {\n",
        "        'feats': torch.cat(all_feats).numpy(),\n",
        "        'targets': torch.cat(all_targets).numpy(),\n",
        "        'preds': torch.cat(all_preds).numpy()\n",
        "    }\n",
        "\n",
        "print(\"Extracting features for COCO Test...\")\n",
        "coco_res = extract_features(coco_loader)\n",
        "print(\"Extracting features for VOC Val...\")\n",
        "voc_res = extract_features(voc_loader)"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Phân tích Các lớp bị nhập nhằng trong COCO\n",
        "Dùng Confusion Matrix để tìm 2 lớp thường bị dự đoán sai thành nhau, và visualize feature của 2 lớp đó."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.metrics import multilabel_confusion_matrix\n",
        "preds_coco_bin = (coco_res['preds'] > 0.5).astype(int)\n",
        "targets_coco = coco_res['targets'].astype(int)\n",
        "\n",
        "mcm = multilabel_confusion_matrix(targets_coco, preds_coco_bin)\n",
        "errors_per_class = mcm[:, 0, 1] + mcm[:, 1, 0] # FP + FN\n",
        "top_err_idx = np.argsort(errors_per_class)[::-1][:5]\n",
        "\n",
        "# Tìm tên lớp tương ứng\n",
        "with open(coco_root + '/annotations/instances_val2017.json') as f:\n",
        "    coco_anns = json.load(f)\n",
        "cats_sorted = sorted(coco_anns['categories'], key=lambda c: c['id'])\n",
        "coco_class_names = [c['name'] for c in cats_sorted]\n",
        "\n",
        "print(\"Top 5 lớp có số lỗi nhiều nhất trong COCO Test:\")\n",
        "for idx in top_err_idx:\n",
        "    print(f\"- {coco_class_names[idx]} (Lỗi: {errors_per_class[idx]})\")\n",
        "    \n",
        "# Chọn 2 lớp có lỗi cao nhất để visualize\n",
        "c1, c2 = top_err_idx[0], top_err_idx[1]\n",
        "print(f\"\\nTrực quan hoá Feature Space của 2 lớp: {coco_class_names[c1]} và {coco_class_names[c2]}\")\n",
        "idx_c1 = np.where(targets_coco[:, c1] == 1)[0]\n",
        "idx_c2 = np.where(targets_coco[:, c2] == 1)[0]\n",
        "idx_both = np.intersect1d(idx_c1, idx_c2)\n",
        "\n",
        "# Loại bỏ các ảnh chứa cả 2 lớp để nhìn rõ sự phân biệt\n",
        "idx_c1_only = np.setdiff1d(idx_c1, idx_both)\n",
        "idx_c2_only = np.setdiff1d(idx_c2, idx_both)\n",
        "\n",
        "feats_c1 = coco_res['feats'][idx_c1_only]\n",
        "feats_c2 = coco_res['feats'][idx_c2_only]\n",
        "\n",
        "X_confused = np.concatenate([feats_c1, feats_c2], axis=0)\n",
        "y_confused = np.array([coco_class_names[c1]] * len(feats_c1) + [coco_class_names[c2]] * len(feats_c2))\n",
        "\n",
        "from sklearn.decomposition import PCA\n",
        "pca_confused = PCA(n_components=2)\n",
        "X_pca_confused = pca_confused.fit_transform(X_confused)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.scatterplot(x=X_pca_confused[:, 0], y=X_pca_confused[:, 1], hue=y_confused, alpha=0.7)\n",
        "plt.title(f'PCA Feature Overlap: {coco_class_names[c1]} vs {coco_class_names[c2]}')\n",
        "plt.show()\n",
        "print(\"Nhận xét: Nếu hai đám mây điểm giao nhau (overlap) nhiều, có nghĩa là mô hình chưa trích xuất được đặc trưng đủ tốt để phân biệt 2 lớp này. \\nLý giải: Bản chất mô hình chiếu vector vào không gian đặc trưng. Nếu chúng chung không gian (overlap) thì W (ma trận trọng số) của classification head không thể phân tách bằng 1 siêu phẳng tuyến tính.\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Domain Shift Analysis (COCO vs Pascal VOC)\n",
        "Việc COCO và VOC là hai tập dữ liệu khác nhau có thể dẫn đến phân bố không gian đặc trưng (Feature Distribution) bị lệch (Domain Shift). Chúng ta sẽ xem xét xem tập VOC có nằm ngoài vùng phân bố của COCO hay không."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Lấy 2000 sample từ COCO và 2000 sample từ VOC để PCA\n",
        "np.random.seed(42)\n",
        "coco_sample_idx = np.random.choice(len(coco_res['feats']), min(2000, len(coco_res['feats'])), replace=False)\n",
        "voc_sample_idx = np.random.choice(len(voc_res['feats']), min(2000, len(voc_res['feats'])), replace=False)\n",
        "\n",
        "X_domain = np.concatenate([coco_res['feats'][coco_sample_idx], voc_res['feats'][voc_sample_idx]], axis=0)\n",
        "y_domain = np.array(['COCO Test'] * len(coco_sample_idx) + ['VOC Val'] * len(voc_sample_idx))\n",
        "\n",
        "pca_domain = PCA(n_components=2)\n",
        "X_pca_domain = pca_domain.fit_transform(X_domain)\n",
        "\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.scatterplot(x=X_pca_domain[:, 0], y=X_pca_domain[:, 1], hue=y_domain, alpha=0.5, s=15, palette=['blue', 'red'])\n",
        "plt.title('Domain Shift Analysis: COCO vs Pascal VOC (PCA)')\n",
        "plt.show()\n",
        "\n",
        "print(\"Nhận xét Domain Shift:\")\n",
        "print(\"- Nếu đám mây đỏ (VOC) bị chệch hẳn so với đám mây xanh (COCO): Có sự dịch chuyển miền (Domain Shift) rõ rệt, dẫn đến mAP giảm mạnh.\")\n",
        "print(\"- Nếu chúng hoà quyện: Domain tương đồng. Lỗi mAP = 0.0 rất có thể chỉ là do Pipeline (ánh xạ nhãn sai, hoặc code đánh giá VOC có lỗi).\")\n"
    ]
})

nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
with open('d:/NA/Kì 6/Thị Giác Máy Tính/Project/ECAAL/notebooks/kaggle_coco_voc_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Notebook created successfully.")
