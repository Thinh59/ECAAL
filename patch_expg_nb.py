import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = None
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for f in files:
        if f == "kaggle-run-exp-g.ipynb":
            nb_path = os.path.join(root, f)
            break

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 0: Update markdown title
nb['cells'][0]['source'] = [
    "# Exp G: EfficientNet-B3 + CBAM + ASL + Strong Augmentation\n",
    "\n",
    "**Cai tien so voi Exp C (EffNet-B0+CBAM+ASL, Test mAP=0.6537):**\n",
    "1. **EfficientNet-B3** (12M params) thay B0 (3.63M) - tang capacity 3.3x cho 80 lop COCO.\n",
    "2. **Dropout 0.5** (tang tu 0.3) - regularize manh hon de CBAM khong overfit tren 16k anh.\n",
    "3. **Strong Augmentation** (RandAugment N=2,M=9 + RandomErasing p=0.25).\n",
    "4. **img_size=300** - resolution toi uu cho EfficientNet-B3.\n",
]

# ── Cell 3: Update config path to new Exp G (index 3 = 4th cell)
EXP_NAME_NEW = 'exp_G_efficientnet_b3_cbam_asl'
nb['cells'][3]['source'] = [
    "## 3. HUAN LUYEN EXP G MOI (EfficientNet-B3 + CBAM + ASL + Strong Augment)\n",
    "EXP_NAME = '%s'\n" % EXP_NAME_NEW,
    "config_path = str(REPO_DIR / 'configs' / (EXP_NAME + '.yaml'))\n",
    "print('Config:', config_path)\n",
    "import os as _os\n",
    "if not _os.path.exists(str(REPO_DIR / 'configs' / (EXP_NAME + '.yaml'))):\n",
    "    print('ERROR: config not found! Make sure you pushed the new config to GitHub.')\n",
    "else:\n",
    "    import subprocess\n",
    "    subprocess.run(['python', str(REPO_DIR / 'src' / 'train.py'), '--config', config_path], check=True)\n",
]

# ── Cell 5: Update EXP_NAME reference
nb['cells'][5]['source'] = [
    "EXP_NAME = '%s'\n" % EXP_NAME_NEW,
    "LOG_FILE = f'/kaggle/working/outputs/{EXP_NAME}/log.json'\n",
    "\n",
    "import os, json, matplotlib.pyplot as plt\n",
    "if os.path.exists(LOG_FILE):\n",
    "    with open(LOG_FILE) as f:\n",
    "        logs = json.load(f)\n",
    "    epochs = [r['epoch'] for r in logs]\n",
    "    train_loss = [r.get('train_loss', 0) for r in logs]\n",
    "    val_map = [r.get('mAP', 0) for r in logs]\n",
    "\n",
    "    fig, ax1 = plt.subplots(figsize=(10, 5))\n",
    "    ax1.plot(epochs, train_loss, 'r-o', label='Train Loss')\n",
    "    ax1.set_xlabel('Epochs')\n",
    "    ax1.set_ylabel('Loss', color='r')\n",
    "    ax1.tick_params(axis='y', labelcolor='r')\n",
    "\n",
    "    ax2 = ax1.twinx()\n",
    "    ax2.plot(epochs, val_map, 'b-o', label='Val mAP')\n",
    "    ax2.set_ylabel('mAP', color='b')\n",
    "    ax2.tick_params(axis='y', labelcolor='b')\n",
    "\n",
    "    lines1, labels1 = ax1.get_legend_handles_labels()\n",
    "    lines2, labels2 = ax2.get_legend_handles_labels()\n",
    "    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')\n",
    "    plt.title('Training Progress - ' + EXP_NAME)\n",
    "    fig.tight_layout()\n",
    "    plt.savefig('/kaggle/working/training_progress.png', dpi=100)\n",
    "    plt.show()\n",
    "    best = max(logs, key=lambda r: r.get('mAP', 0))\n",
    "    print('Best epoch: %d | Val mAP: %.4f | Macro-F1: %.4f' % (best['epoch'], best['mAP'], best.get('macro_f1', 0)))\n",
    "else:\n",
    "    print('Log file not found:', LOG_FILE)\n",
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved notebook OK')
print('Cell 3 first line:', nb['cells'][3]['source'][0])
