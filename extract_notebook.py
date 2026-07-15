import json

with open('notebooks/diamond_dynamics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print(''.join(cell['source']))
        print('# ---')
