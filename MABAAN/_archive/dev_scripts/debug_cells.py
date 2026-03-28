"""Show problematic cells."""
import json

nb = json.load(open(r'notebooks/MABAAN_v4_Main.ipynb', encoding='utf-8'))
for i in [2, 3, 6, 7, 20]:
    cell = nb['cells'][i]
    if cell['cell_type'] != 'code':
        print(f"Cell {i} is markdown, skip")
        continue
    src = ''.join(cell['source'])
    print(f"\n--- Cell {i} (first 5 lines) ---")
    for j, line in enumerate(src.split('\n')[:5]):
        print(f"  {j}: {repr(line)}")
