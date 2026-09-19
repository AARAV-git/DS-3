import json

data = json.load(open('logs/history_v4.json'))
print(f'Epochs run: {len(data)}')
best = max(data, key=lambda x: x['val_origin_acc'])
print(f'Best VlOAcc: {best["val_origin_acc"]*100:.2f}% at Epoch {best["epoch"]}')
pc = best['per_class_acc']
print(f'  ai={pc[0]:.1f}%  edit={pc[1]:.1f}%  real={pc[2]:.1f}%')
print()
print(f'Ep | TrOAcc | VlOAcc | gap   | ai    | edit  | real')
print('-' * 55)
for ep in data:
    pc = ep['per_class_acc']
    gap = ep['train_origin_acc'] - ep['val_origin_acc']
    print(f'{ep["epoch"]:>2} | {ep["train_origin_acc"]*100:5.1f}% | {ep["val_origin_acc"]*100:5.1f}% | {gap*100:5.1f}% | {pc[0]:5.1f}% | {pc[1]:5.1f}% | {pc[2]:5.1f}%')

