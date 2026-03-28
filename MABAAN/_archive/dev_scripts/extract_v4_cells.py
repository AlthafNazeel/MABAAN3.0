"""Extract cell sources from MABAAN_V4_Main_1.ipynb for analysis."""
import json

nb = json.load(open(r'notebooks/MABAAN_V4_Main_1.ipynb', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}\n")

for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    lines = src.strip().split('\n')
    preview = lines[0][:100] if lines else '(empty)'
    n_lines = len(lines)
    exec_ct = cell.get('execution_count', '-')
    print(f"Cell {i:2d} [{ct:>8}] exec={exec_ct} lines={n_lines:3d} | {preview}")
