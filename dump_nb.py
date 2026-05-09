import json, io, os, sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = None
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for f in files:
        if f == "kaggle-run-exp-g.ipynb":
            nb_path = os.path.join(root, f)
            break

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

out = io.StringIO()
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if src.strip():
        ct = c['cell_type']
        out.write('---CELL %d (%s)---\n' % (i, ct))
        out.write(src[:800])
        out.write('\n\n')

dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb_dump.txt")
with open(dump_path, 'w', encoding='utf-8') as o:
    o.write(out.getvalue())
print('Done')
