"""Extract error outputs from the executed Kaggle notebook."""
import json

nb = json.load(open(r'd:\Althaf\IIT\Final Year\FYP\Implementation Workspace\MABAAN\MABAAN_V4_Main_Kaggle.ipynb', encoding='utf-8'))

for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    src_preview = ''.join(cell['source'][:2])[:80] if cell['source'] else ''
    
    # Check for errors in outputs
    if cell_type == 'code' and 'outputs' in cell:
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                print(f"\n=== ERROR in Cell {i} ===")
                print(f"Source preview: {src_preview}")
                print(f"Error: {out.get('ename', 'Unknown')}: {out.get('evalue', '')}")
                # Print traceback
                for line in out.get('traceback', [])[-5:]:
                    # Strip ANSI codes
                    import re
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    print(clean)

    # Also show which cells executed successfully (last few)
    if cell_type == 'code' and cell.get('execution_count'):
        has_error = any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
        status = "ERROR" if has_error else "OK"
        if cell.get('execution_count', 0) > 15 or has_error:
            print(f"Cell {i} (exec #{cell.get('execution_count')}): {status} | {src_preview}")
