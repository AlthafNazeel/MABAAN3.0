"""Dump all code cells to file."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

nb = json.load(open(r'notebooks/MABAAN_V4_Main_1.ipynb', encoding='utf-8'))

with open('cell_dump.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        f.write(f"\n{'='*60}\n")
        f.write(f"CELL {i} ({len(src.split(chr(10)))} lines)\n")
        f.write(f"{'='*60}\n")
        f.write(src + "\n")

print("Written to cell_dump.txt")
