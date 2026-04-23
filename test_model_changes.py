import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(r'd:\NA\Kì 6\Thị Giác Máy Tính\Project\ECAAL\src')))
from models import build_model

def test_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"\n--- Testing {cfg_path} ---")
    model = build_model(cfg['model'])
    
    # Test forward
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
        print("Output shape:", out.shape)
        
    # Test train mode (with masking if applicable)
    model.train()
    out = model(x)
    print("Train forward passed.")

test_config(r'd:\NA\Kì 6\Thị Giác Máy Tính\Project\ECAAL\configs\exp_G_masked_attention.yaml')
test_config(r'd:\NA\Kì 6\Thị Giác Máy Tính\Project\ECAAL\configs\exp_H_consistency.yaml')
test_config(r'd:\NA\Kì 6\Thị Giác Máy Tính\Project\ECAAL\configs\exp_I_sparsity.yaml')
