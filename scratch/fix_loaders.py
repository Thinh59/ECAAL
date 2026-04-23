import json
from pathlib import Path

def fix_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'build_dataloaders' in source:
                new_source = [
                    "from dataset import COCOMultiLabelDataset, get_val_transform, get_train_transform\n",
                    "from torch.utils.data import DataLoader\n",
                    "import json\n",
                    "test_ids_file = SUBSET_DIR / 'subset_test_ids.json'\n",
                    "test_ids = json.load(open(test_ids_file)) if test_ids_file.exists() else None\n",
                    "test_ds = COCOMultiLabelDataset(coco_root, split='val', transform=get_val_transform(224), subset_ids=test_ids)\n",
                    "test_loader = DataLoader(test_ds, batch_size=32, num_workers=2, shuffle=False)\n"
                ]
                # If it's deep analysis, we also need train_loader
                if 'train_loader' in source:
                    new_source.extend([
                        "train_ids_file = SUBSET_DIR / 'subset_train_ids.json'\n",
                        "train_ids = json.load(open(train_ids_file)) if train_ids_file.exists() else None\n",
                        "train_ds = COCOMultiLabelDataset(coco_root, split='train', transform=get_train_transform(224), subset_ids=train_ids)\n",
                        "train_loader = DataLoader(train_ds, batch_size=32, num_workers=2, shuffle=False)\n"
                    ])
                cell['source'] = new_source
                print(f"Fixed {nb_path.name}")
                
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

nb_dir = Path('d:/NA/Kì 6/Thị Giác Máy Tính/Project/ECAAL/notebooks')
fix_notebook(nb_dir / 'kaggle_deep_analysis.ipynb')
fix_notebook(nb_dir / 'kaggle_cbam_small_objects.ipynb')
print("Xong!")
