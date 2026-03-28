"""Verify all v4 notebooks for syntax correctness."""
import json, sys

notebooks = [
    'notebooks/MABAAN_v4_Main.ipynb',
    'notebooks/MABAAN_v4_Baseline.ipynb',
    'notebooks/MABAAN_v4_Ablation.ipynb',
]

all_ok = True
for path in notebooks:
    try:
        nb = json.load(open(path, encoding='utf-8'))
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
        print(f"{path}: {len(nb['cells'])} cells ({len(code_cells)} code, {len(md_cells)} markdown)")
        
        for i, cell in enumerate(code_cells):
            src = '\n'.join(cell['source'])
            # Skip cells starting with ! (shell commands)
            if src.strip().startswith('!'):
                continue
            try:
                compile(src, f'<{path}:cell_{i}>', 'exec')
            except SyntaxError as e:
                print(f"  SYNTAX ERROR in code cell {i}: {e}")
                print(f"  Near: {e.text}")
                all_ok = False
        
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        all_ok = False

print(f"\n{'ALL OK' if all_ok else 'ERRORS FOUND'}")
sys.exit(0 if all_ok else 1)
