"""More detailed extraction - ASCII only."""
import json, re

nb = json.load(open(r'd:\Althaf\IIT\Final Year\FYP\Implementation Workspace\MABAAN\MABAAN_V4_Main_Kaggle.ipynb', encoding='utf-8'))

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    src_line1 = src.strip().split('\n')[0][:90] if src.strip() else '(empty)'
    exec_count = cell.get('execution_count', None)
    outputs = cell.get('outputs', [])
    
    if ct == 'markdown':
        status = "MD"
    elif exec_count is not None:
        status = f"exec#{exec_count}"
    else:
        status = "NOT_RUN"
    
    has_error = any(o.get('output_type') == 'error' for o in outputs)
    marker = "ERR" if has_error else ("---" if status == "NOT_RUN" else " OK")
    
    print(f"[{marker}] Cell {i:2d} [{status:>8}] {src_line1}")
    if has_error:
        for out in outputs:
            if out.get('output_type') == 'error':
                print(f"    ERROR: {out.get('ename')}: {out.get('evalue', '')[:300]}")
                tb = out.get('traceback', [])
                for line in tb[-4:]:
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    for cl in clean.split('\n'):
                        cl = cl.strip()
                        if cl:
                            print(f"    TB: {cl[:200]}")

print("\n=== Training cell output (last lines) ===")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'tracker = train_model' in src:
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
                lines = text.strip().split('\n')
                print(f"Cell {i}, last 20 lines:")
                for line in lines[-20:]:
                    print(f"  {line}")

print("\n=== Eval cell source and output ===")                    
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'run_inference' in src and 'evaluate_model' in src:
        print(f"Cell {i} source:")
        print(src[:500])
        print("---outputs---")
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                print(f"ERROR: {out['ename']}: {out['evalue'][:300]}")
                tb = out.get('traceback', [])
                for line in tb[-5:]:
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    print(clean[:300])
            elif out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
                print(text[:500])
