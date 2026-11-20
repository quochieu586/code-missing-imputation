"""Exploratory Tier 3 adapter; frozen scoring remains in tier3.py."""
import argparse
import csv
from pathlib import Path
import numpy as np
from longitudinal_impute.data import load_csv, model_input, restore_counts
from longitudinal_impute.evaluation import tier1, tier2, tier3 as t
from longitudinal_impute.baselines.mean_fill import MeanFill
from longitudinal_impute.baselines.zero_fill import ZeroFill
from longitudinal_impute.baselines.marginal_sampler import MarginalSampler

OUT = Path(__file__).resolve().parent
t.REAL = ('hron', 'tsagris', 'micoda', 'same_day_mean', 'main', 'main_nometa')
t.MODELS = t.REAL + tuple(t.DECOYS)
original_decoy_scores = t.decoy_worse
def extended_decoy_scores(scores, decoy, real=None, higher_is_worse=True):
    return original_decoy_scores(scores, decoy, real=t.REAL if real is None else real,
                                 higher_is_worse=higher_is_worse)
t.decoy_worse = extended_decoy_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', nargs='+', default=['whole_feature', 'random_cell', 'whole_feature_matched'])
    args = parser.parse_args()
    dest = OUT / 'report_tier3'
    dest.mkdir(exist_ok=True)
    data = load_csv(tier2.DATA, tier2.SCHEMA)
    facts = tier2.row_facts(data)
    cases, diagnostic, rates, ranks, decoys, headline = {}, [], [], [], [], []
    verdict_rows = []
    md = ['# Tier 3 â€” exploratory', '',
          'Sai lá»‡ch giao thá»©c: seed 300â€“309 thay 0â€“4; thĂªm main (metadata 4 trÆ°á»ng) vĂ  main_nometa. KhĂ´ng Ă¡p dá»¥ng cá»•ng a17/tĂªn run/git sáº¡ch. Scorer vĂ  ngÆ°á»¡ng giá»¯ nguyĂªn.', '',
          'U1: thá»© háº¡ng cháº©n Ä‘oĂ¡n cĂ³ khá»›p sai sá»‘ tháº­t? U2: chá»n theo cháº©n Ä‘oĂ¡n cĂ³ gáº§n model tá»‘t nháº¥t? U3: tĂ­n hiá»‡u ngoĂ i khoáº£ng thá»i gian cĂ³ phĂ¢n biá»‡t Ă´ dÆ°Æ¡ng? U4: decoy cĂ³ bá»‹ Ä‘Ă¡nh giĂ¡ tá»‡ hÆ¡n model tháº­t?', '',
          '; '.join(f'{k}={getattr(t,k)}' for k in ('TIE','PASS_RATE','MIN_CASE_RATE','REGRET_MAX','U3_GAP','GOOD_CASES')), '']
    for mask in args.masks:
        folder = OUT / mask
        status, scores, _, _ = tier1.load_run(folder, {}, mask)
        runs = {m:{int(s):r for s,r in v.items()} for m,v in tier1.table_runs(status, scores).items() if m in t.REAL}
        cases[mask] = []
        for seed in range(300,310):
            hidden = np.load(folder / f'mask_{mask}_seed{seed}.npy')
            observed = model_input(data, hidden)
            preds = {m:np.load(folder / f'predictions/{mask}_seed{seed}_{m}.npy') for m in t.REAL}
            for name, model in [('mean_fill',MeanFill()), ('zero_fill',ZeroFill()), ('marginal_sampler',MarginalSampler(seed=seed))]:
                model.fit(observed)
                counts = restore_counts(data, model.predict(observed), hidden)
                preds[name] = counts
                error = t.errors_on(data.counts, preds[name], data.totals, tier2.calibration_cells(data, hidden)['eval'])
                runs.setdefault(name,{})[seed] = {'eligible':True,'mae_prop':error['all']['mae_proportion']}
            c, cells, support = t.one_case(data, facts, hidden, {(m,seed):v for m,v in preds.items()}, runs, seed)
            cases[mask].append(c)
            for m in t.MODELS:
                error = runs[m][seed]['mae_prop']
                headline.append(dict(mask=mask,seed=seed,model=m,mae_prop=error))
                for d, value in c['diag'][m].items():
                    diagnostic.append(dict(mask=mask,seed=seed,model=m,diagnostic=d,value=value,true_error_mae_prop=error,kind='overview'))
            for ref in ('local', 'pooled'):
                blocks = c['u'][ref]
                for d in ('d1','d2'):
                    for m, value in blocks[d]['dist'].items():
                        for group, err in blocks[d]['err'][m].items():
                            diagnostic.append(dict(mask=mask,seed=seed,model=m,reference=ref,diagnostic=d,value=value,true_error_mae_prop=err['mae_proportion'],group=group,kind='scored_U2' if d=='d2' and group=='all' else 'scored_scalar'))
                for key in ('u1_under','u1_over','u2','u2_raw','u2_full'):
                    row=dict(mask=mask,seed=seed,reference=ref,case=key,**blocks[key])
                    (rates if key.startswith('u1') else ranks).append(row)
                for key in ('regret','regret_raw','regret_full'):
                    ranks.append(dict(mask=mask,seed=seed,reference=ref,case=key,value=blocks[key]))
            rates.append(dict(mask=mask,seed=seed,case='u3',**c['u3']))
            for name,v in c['u4'].items():
                for model, value in v.items():
                    score = c['u4_scores'][name].get(model)
                    real_scores = [c['u4_scores'][name].get(m) for m in t.REAL]
                    rank = None if score is None else 1 + sum(s < score for s in real_scores if s is not None)
                    decoys.append(dict(mask=mask,seed=seed,diagnostic=name,model=model,worse_than_all_real=value,score=score,rank_among_real_plus_decoy=rank,real_models_available=sum(s is not None for s in real_scores)))
        md += ['## '+mask+' â€” exploratory','']
        for fun in (t.u1_table,t.u2_table,t.u3_table):
            lines, verdict = fun({mask:cases[mask]})
            md += lines + ['']
            for key, val in (verdict.items() if isinstance(verdict, dict) else [(fun.__name__.split('_')[0], verdict)]):
                verdict_rows.append(dict(mask=mask,case=key,verdict=val))
            print(mask,fun.__name__,verdict,flush=True)
        md += t.sensitivity_lines({mask:cases[mask]})[0] + [''] + [line.replace('qua 5 seed', 'qua 10 seed') for line in t.u4_table({mask:cases[mask]})]
        verdict_rows.append(dict(mask=mask,case='u4',verdict='descriptive_no_frozen_threshold'))
    md += ['', '## Kết luận chính — exploratory', '',
           'U4 chỉ mô tả trong scorer gốc, không có ngưỡng pass/fail đóng băng; không tự đặt thêm ngưỡng. Zero-fill có D3 bằng 0 nhưng sai số lớn: D3 không dùng riêng để chọn model.', '',
           '| Mask | U-case | Verdict |', '|---|---|---|']
    md += [f"| {r['mask']} | {r['case']} | {r['verdict']} |" for r in verdict_rows]
    md += ['', 'MAE prop toàn ô model (mean qua 10 seed; thấp hơn tốt hơn):', '',
           '| Mask | main | main_nometa | Baseline tốt nhất | MAE baseline |', '|---|---:|---:|---|---:|']
    for mask in args.masks:
        means = {m:float(np.mean([r['mae_prop'] for r in headline if r['mask']==mask and r['model']==m])) for m in t.REAL}
        best = min(t.REAL[:4], key=means.get)
        md.append(f"| {mask} | {means['main']:.6f} | {means['main_nometa']:.6f} | {best} | {means[best]:.6f} |")
    md += ['', 'CSV: diagnostic_vs_error chứa giá trị tổng quan D1/D2/D3 và scalar R-local/R-pooled dùng cho U1/U2; kind=scored_U2 là |D2−D2_truth| đối chiếu MAE prop trên target chung. rates chứa U1/U3; rank_agreement_regret chứa khớp hạng và regret theo seed; u4_decoy_ranks: hạng 1 là tốt nhất, so decoy với sáu model thật; NA được giữ trống. Decoy sinh bằng baseline gốc trên dữ liệu đã che, seed tương ứng, restore_counts như runner.', '']
    for name,rows in [('diagnostic_vs_error',diagnostic),('rates',rates),('rank_agreement_regret',ranks),('u4_decoy_ranks',decoys),('headline_mae',headline),('verdicts',verdict_rows)]:
        with (dest/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader(); w.writerows(rows)
    (dest/'REPORT_TIER3_VI.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()












