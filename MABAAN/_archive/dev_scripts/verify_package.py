"""Verify all mabaan package modules compile cleanly."""
import py_compile
import os

errors = []
for root, dirs, files in os.walk('mabaan'):
    for fn in files:
        if fn.endswith('.py'):
            fpath = os.path.join(root, fn)
            try:
                py_compile.compile(fpath, doraise=True)
                print(f"  OK  {fpath}")
            except py_compile.PyCompileError as e:
                print(f" ERR  {fpath}: {e}")
                errors.append(fpath)

# Also verify notebook code cells
import json
nb = json.load(open(r'notebooks/MABAAN_v4_Main.ipynb', encoding='utf-8'))
code_ok = True
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if src.startswith('!'):  # skip pip install
        continue
    try:
        compile(src, f'cell_{i}', 'exec')
    except SyntaxError as e:
        print(f" ERR  Notebook cell {i}: {e}")
        code_ok = False
        errors.append(f"cell_{i}")

if errors:
    print(f"\n{len(errors)} errors found!")
else:
    print(f"\nALL OK - Package + Notebook verified")
