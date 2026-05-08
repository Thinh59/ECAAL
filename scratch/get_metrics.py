import json
from pathlib import Path

base = Path('results/ecaal_train_val_results/outputs')

exps = {
    'A': 'exp_A_resnet_bce',
    'B': 'exp_B_resnet_asl',
    'C': 'exp_C_efficientnet_cbam_asl',
    'D': 'exp_D_resnet_focal',
    'E': 'exp_E_resnet_cbam_asl',
    'F': 'exp_F_efficientnet_asl',
    'G': 'exp_G',
}

print('Val mAP and Macro-F1 at best epoch (from log.json):')
print(f"{'Exp':<5} {'Dir':<35} {'Val_mAP':>10} {'Val_F1':>10} {'Best_Epoch':>12} {'Train_Loss':>12}")
print('-'*90)
for name, d in exps.items():
    log_path = base / d / 'log.json'
    if not log_path.exists():
        print(f'{name:<5} {d:<35} NOT FOUND')
        continue
    records = json.loads(log_path.read_text(encoding='utf-8'))
    best = max(records, key=lambda r: r.get('mAP', 0))
    print(f"{name:<5} {d:<35} {best.get('mAP', 0):>10.4f} {best.get('macro_f1', 0):>10.4f} {best.get('epoch', '?'):>12} {best.get('train_loss', 0):>12.4f}")
