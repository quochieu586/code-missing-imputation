"""Read existing v2 public masks only; export a fixed crop as editable TikZ.

No generator, masking backend, fitted model or benchmark is imported/executed.
Selection: easy profile, lexicographically first entity, central 16 of 120 times,
all 12 named components. Selection does not inspect prediction errors.
"""
from pathlib import Path
import csv, hashlib, json

ROOT = Path(__file__).resolve().parents[3]
DECK = ROOT / 'presentation/implementation_report'
BUNDLE = ROOT / 'datasets/synthetic/wave_panel_v2/easy'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[]; sources={}; scopes=[]; meta=[]
    for name, origin in [('01_mcar_declared_patterns',0.85),('02_mar_declared_patterns',6.05)]:
        folder=BUNDLE/'conditioning'/name
        with (folder/'conditioning.csv').open(encoding='utf-8-sig',newline='') as f: panel=list(csv.DictReader(f))
        with (folder/'hidden_mask.csv').open(encoding='utf-8',newline='') as f: mask=[[int(x) for x in row] for row in csv.reader(f)]
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert len(panel)==len(mask)==7200 and all(len(r)==12 for r in mask)
        entity=min(r['entity_id'] for r in panel)
        ids=sorted([i for i,r in enumerate(panel) if r['entity_id']==entity],key=lambda i:panel[i]['date'])
        assert len(ids)==120
        start=(len(ids)-16)//2
        crop=ids[start:start+16]
        lines=[f'% {name}: {entity}, indices {start}..{start+15}; saved hidden_mask.csv only']
        for t, i in enumerate(crop):
            for j, hidden in enumerate(mask[i]):
                x=origin+t*.22; top=3.35-j*.14; y=top-.14
                lines.append(f'\\draw[draw=Gridline,line width=0.25pt,fill=BKLight!65] ({x:.3f},{y:.3f}) rectangle ({x+.22:.3f},{top:.3f});')
                if hidden:
                    lines.append(f'\\draw[MicodaOrange,line width=0.65pt] ({x+.05:.3f},{y+.03:.3f})--({x+.17:.3f},{top-.03:.3f}) ({x+.05:.3f},{top-.03:.3f})--({x+.17:.3f},{y+.03:.3f});')
                records.append({'mechanism':manifest['spec']['mechanism'],'entity_id':entity,'time_index':start+t,'date':panel[i]['date'],'component':f'component_{j+1:02d}','hidden':hidden})
        scopes.extend(lines)
        meta.append({'mechanism':manifest['spec']['mechanism'],'entity_id':entity,'time_indices':[start,start+15],'dates':[panel[i]['date'] for i in crop],'components':'component_01..component_12','dataset_seed':2026,'mask_seed':manifest['spec']['seed'],'full_panel_hidden_cells':manifest['diagnostics']['hidden_target_cells'],'full_panel_target_cells':manifest['diagnostics']['target_cells'],'crop_hidden_cells':sum(sum(mask[i]) for i in crop)})
        for file in ('conditioning.csv','hidden_mask.csv','manifest.json'):
            p=folder/file;sources[p.relative_to(ROOT).as_posix()]=sha(p)
    tex=DECK/'figures/ticket02_missingness_masks.tex';tex.write_text('\n'.join(scopes)+'\n',encoding='utf-8')
    out=DECK/'data/ticket02_missingness_crop.csv'
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    provenance={'selection_rule':__doc__,'crop':meta,'source_sha256':sources,'outputs':{tex.relative_to(ROOT).as_posix():sha(tex),out.relative_to(ROOT).as_posix():sha(out)}}
    (DECK/'data/ticket02_missingness_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
