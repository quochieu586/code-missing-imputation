"""CPU-friendly, reproducible single-fold pilot. No outer diffusion loop.

Run: python scripts/run_stage_b.py --epochs 100 --samples 5
Actual missing cells have no ground truth. Evaluation uses complete held-out
location/date rows, with targets removed BEFORE preprocessing and KNN.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.core.transforms import clr
from src.stage_a.knn_aitchison import knn_aitchison_impute
from src.stage_b.csdi.model import Denoiser
from src.stage_b.single_pass import training_loss, sample, single_pass
from src.evaluation.metrics import evaluate


def windows(df, length):
    result = []
    for _, group in df.groupby('location', sort=True):
        idx = group.sort_values('date').index.to_numpy()
        for start in range(0, len(idx), length):
            part = idx[min(start, max(0, len(idx)-length)):][:length]
            result.append(part)
    return result


def batches(parts, size):
    # Bucket by length, avoiding padding in the time transformer.
    for length in sorted(set(map(len, parts))):
        group = [p for p in parts if len(p) == length]
        for start in range(0, len(group), size):
            yield np.stack(group[start:start+size])


def predict(model, x, mask, times, parts, beta, samples, batch_size):
    out = np.zeros_like(x)
    for idx in batches(parts, batch_size):
        z = sample(model, torch.tensor(x[idx], dtype=torch.float32),
                   torch.tensor(mask[idx]), torch.tensor(times[idx], dtype=torch.float32), beta, samples)
        out[idx.reshape(-1)] = z.numpy().reshape(-1, x.shape[1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--fold', type=int, default=0, choices=range(5))
    parser.add_argument('--checkpoint', help='Reuse a trained checkpoint; sampling/audit only, no retraining')
    parser.add_argument('--evaluation-cache', help='Reuse matching audited KNN arrays (source hash/config are verified)')
    parser.add_argument('--output', default='artifacts/stage_b_single_pass_final')
    args = parser.parse_args()
    if min(args.epochs,args.samples,args.steps,args.window,args.batch_size) < 1:
        parser.error('training and sampling parameters must be positive')
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path('data/covariants.csv')
    df = pd.read_csv(source)
    df['date'] = pd.to_datetime(df.date)
    cols = [c for c in df if c not in ['location','date','total_sequence']]
    raw = df[cols].to_numpy(float)
    m0 = np.isfinite(raw)
    m0.setflags(write=False)
    total = df.total_sequence.to_numpy(float)[:,None]
    filtered = raw.copy()
    filtered[(raw >= 0) & (total > 0) & (raw/np.maximum(total,1) < 1e-4)] = 0
    zero = filtered == 0
    subjects = np.array(sorted(df.location.unique()))
    rng.shuffle(subjects)
    complete = m0.all(1)
    complete_subjects = df.loc[complete,'location'].unique()
    # Balance subjects with evaluable compositions across folds, using only
    # missingness metadata (never abundance values or resulting errors).
    strata_a = subjects[np.isin(subjects,complete_subjects)]
    strata_b = subjects[~np.isin(subjects,complete_subjects)]
    test_subjects = np.concatenate([np.array_split(strata_a,5)[args.fold],
                                    np.array_split(strata_b,5)[args.fold]])
    test_rows = df.location.isin(test_subjects).to_numpy()
    train_rows = ~test_rows
    # Benchmark: complete compositions only, so true CLR is identifiable.
    test_mask = (rng.random(raw.shape) < .5) & (test_rows & complete)[:,None]
    val_mask = (rng.random(raw.shape) < .1) & (train_rows & complete)[:,None]
    available = m0 & ~test_mask & ~val_mask
    available.setflags(write=False)
    positives = filtered[available & train_rows[:,None] & (filtered > 0)]
    pseudo = float(positives.min()/2)
    truth = np.where(zero, pseudo, filtered)
    observed = np.where(available, truth, np.nan)
    train_idx = np.flatnonzero(train_rows)
    # k is a fixed inherited value, not tuned on test data.
    print('Preparing train-only KNN for evaluation', flush=True)
    if args.evaluation_cache:
        cachedir = Path(args.evaluation_cache)
        cached_record = json.loads((cachedir/'results.json').read_text(encoding='utf-8'))
        assert cached_record['data_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        assert cached_record['config']['seed']==args.seed and cached_record['config']['fold']==args.fold
        cached = np.load(cachedir/'audit_arrays.npz')
        assert np.array_equal(cached['available'],available)
        assert np.array_equal(cached['train_rows'],train_rows)
        assert np.array_equal(cached['truth'],truth,equal_nan=True)
        x1 = cached['evaluation_X1']
        n_fallback = cached_record['knn_fallback']
    else:
        knn = knn_aitchison_impute(observed, available, k=8, train_idx=train_idx)
        x1 = knn.X_imputed
        n_fallback = knn.n_fallback
    medians = torch.tensor(np.nanmedian(observed[train_idx],axis=0),dtype=torch.float32)
    assert torch.isfinite(medians).all()
    times = (df.date-df.loc[train_rows,'date'].min()).dt.days.to_numpy(float)/14
    parts = windows(df, args.window)
    train_parts = [p for p in parts if train_rows[p[0]]]
    val_parts = [p for p in train_parts if val_mask[p].any()]
    model = Denoiser(len(cols), args.channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    beta = torch.linspace(1e-4**.5, .5**.5, args.steps).square()
    alpha = torch.cumprod(1-beta,0)
    losses = []
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert checkpoint['features']==cols
        assert checkpoint['args']['seed']==args.seed and checkpoint['args']['fold']==args.fold
        assert checkpoint['args']['channels']==args.channels and checkpoint['args']['steps']==args.steps
        assert set(checkpoint['train_subjects'])==set(subjects[~np.isin(subjects,test_subjects)])
        model.load_state_dict(checkpoint['state_dict'])
        previous = json.loads(Path(args.checkpoint).with_name('results.json').read_text(encoding='utf-8'))
        losses = previous['loss_history']
    for epoch in range(0 if args.checkpoint else args.epochs):
        model.train()
        rng.shuffle(train_parts)
        values = []
        for idx in batches(train_parts,args.batch_size):
            optimizer.zero_grad()
            loss = training_loss(model, torch.tensor(x1[idx],dtype=torch.float32),
                                 torch.tensor(available[idx]), torch.tensor(times[idx],dtype=torch.float32),
                                 medians, alpha)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            values.append(loss.item())
        losses.append(float(np.mean(values)))
        print(f'Epoch {epoch+1}/{args.epochs}: loss={losses[-1]:.6f}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    # Fixed training budget: validation is diagnostic, no test-based selection.
    print('Sampling evaluation once',flush=True)
    torch.manual_seed(args.seed+1)
    eval_parts = [p for p in parts if test_mask[p].any() or val_mask[p].any()]
    z = predict(model,x1,available,times,eval_parts,beta,args.samples,args.batch_size)
    # Rows outside sampled windows retain their initializer.
    covered = np.unique(np.concatenate(eval_parts))
    z_full = clr(x1)
    z_full[covered] = z[covered]
    result = single_pass(x1,available,z_full)
    test_complete = test_rows & complete
    val_complete = train_rows & complete & val_mask.any(1)
    metrics = {}
    # Cheap baselines use the identical masks and training-only statistics.
    mean = observed.copy()
    mean_values = np.nanmean(observed[train_idx],axis=0)
    mean[~available] = np.broadcast_to(mean_values,mean.shape)[~available]
    locf, linear = observed.copy(), observed.copy()
    for _, group in df.groupby('location'):
        idx = group.sort_values('date').index.to_numpy()
        for j in range(len(cols)):
            valid = available[idx,j]
            if valid.any():
                linear[idx,j] = np.interp(times[idx], times[idx][valid], observed[idx,j][valid])
                locf[idx,j] = pd.Series(observed[idx,j]).ffill().fillna(mean_values[j]).to_numpy()
            else:
                linear[idx,j] = locf[idx,j] = mean_values[j]
    for name, pred in [('mean',mean),('locf',locf),('linear',linear),('knn_aitchison',x1),('knn_csdi_one_pass',result.X_hat)]:
        metrics[name] = evaluate(truth[test_complete],pred[test_complete],test_mask[test_complete],zero[test_complete])
    validation = evaluate(truth[val_complete],result.X_hat[val_complete],val_mask[val_complete],zero[val_complete])
    validation_knn = evaluate(truth[val_complete],x1[val_complete],val_mask[val_complete],zero[val_complete])
    result.c3_history = [validation['mae_clr_all']]
    # Production continuation uses the existing Stage A artifact, with all original
    # observations restored. It is NEVER used as the evaluation initializer.
    saved = pd.read_csv('data/processed/X1_knn_imputed.csv')
    assert saved.location.equals(df.location)
    assert np.array_equal(pd.to_datetime(saved.date).to_numpy(),df.date.to_numpy())
    production_x1 = saved[cols].to_numpy(float)
    assert np.array_equal(production_x1[m0],truth[m0]), 'Stage A preprocessing differs'
    bridge_frame = df[['location','date']].copy()
    bridge_frame[cols] = clr(production_x1)
    bridge_frame.to_csv(output/'Z1_clr.csv',index=False)
    print('Sampling original missing cells from the existing Stage A artifact',flush=True)
    torch.manual_seed(args.seed+2)
    production_z = predict(model,production_x1,m0,times,parts,beta,args.samples,args.batch_size)
    production = single_pass(production_x1,m0,production_z,validation_mae=validation['mae_clr_all'])
    completed = df.copy()
    completed[cols] = production.X_hat
    completed.to_csv(output/'X2_diffusion_imputed.csv',index=False)
    clr_frame = df[['location','date']].copy()
    clr_frame[cols] = clr(production.X_hat)
    clr_frame.to_csv(output/'Z2_clr.csv',index=False)
    np.savez_compressed(output/'audit_arrays.npz',M0=m0,available=available,test_mask=test_mask,
                        validation_mask=val_mask,train_rows=train_rows,test_rows=test_rows,
                        evaluation_X1=x1,evaluation_X2=result.X_hat,truth=truth,
                        production_Z1=clr(production_x1),production_Z2=clr(production.X_hat))
    torch.save({'state_dict':model.state_dict(),'args':vars(args),'features':cols,
                'train_subjects':subjects[~np.isin(subjects,test_subjects)].tolist(),
                'pseudo_count':pseudo},output/'model.pt')
    record = dict(config=vars(args),max_iter=1,n_iter=1,converged=False,
                  data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  pseudo_count=pseudo,train_subjects=int(len(subjects)-len(test_subjects)),
                  test_subjects=test_subjects.tolist(),test_complete_rows=int(test_complete.sum()),
                  evaluated_subjects=df.loc[test_complete,'location'].unique().tolist(),
                  train_complete_rows=int((complete & train_rows).sum()),knn_fallback=n_fallback,
                  metrics=metrics,validation=validation,validation_knn=validation_knn,loss_history=losses,
                  evaluation_outcome=('worse_than_knn' if metrics['knn_csdi_one_pass']['mae_clr_all'] > metrics['knn_aitchison']['mae_clr_all'] else 'lower_mae_than_knn_in_this_pilot'),
                  training_reused=bool(args.checkpoint),
                  diagnostics=dict(c1_history=production.c1_history,c2_history=production.c2_history,
                                   c3_history=production.c3_history,alpha_div_history=production.alpha_div_history,
                                   zero_prop_history=production.zero_prop_history),
                  observed_preserved=bool(np.array_equal(production.X_hat[m0],production_x1[m0])),
                  elapsed_seconds=time.perf_counter()-started)
    (output/'results.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(metrics).T.to_csv(output/'metrics.csv')
    report = ['# CLR + CSDI: kết quả một lượt', '',
              f'Chạy CPU; {args.epochs} epochs, {args.steps} bước DDPM, {args.samples} mẫu; max_iter=1, n_iter=1, converged=False.',
              f'Fold {args.fold}/5, train {len(subjects)-len(test_subjects)} địa điểm, test {len(test_subjects)} địa điểm.',
              f'Chấm {test_complete.sum()} hàng test đầy đủ trước che; MCAR 50% theo ô. Pseudo-count train-only: {pseudo}.', '',
              'Địa điểm có ground truth đủ để chấm: '+', '.join(df.loc[test_complete,'location'].unique())+'.', '',
              '| Phương pháp | MAE CLR | MAE khác 0 | MAE pseudo-count | M2 | M3 |',
              '|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        report.append('| '+name+' | '+' | '.join(f'{m[key]:.6f}' if m[key] is not None else 'NA' for key in ['mae_clr_all','mae_clr_nonzero','mae_clr_pseudocount','m2','m3'])+' |')
    improvement = 100*(1-metrics['knn_csdi_one_pass']['mae_clr_all']/metrics['knn_aitchison']['mae_clr_all'])
    report += ['',f'Thay đổi MAE so với KNN: {improvement:+.2f}% (dương = giảm lỗi).',
               f'Validation MAE CLR: {validation["mae_clr_all"]:.6f}. Thời gian: {record["elapsed_seconds"]:.1f} giây.',
               '', '## Phạm vi và giới hạn',
               '- Đây là pilot một fold, không phải kết quả mean (SD) của 5-fold CV.',
               '- Dữ liệu SARS-CoV-2 không có phylum map. Dùng CSDI thích ứng; chưa đánh giá CSDI+phylum CNN. Không suy diễn taxonomy.',
               '- Model nhỏ: một transformer thời gian và một transformer đặc trưng, cửa sổ 16 mốc (hoặc cấu hình --window). Không phải tái lập nguyên cấu hình bài báo.',
               '- Không chấm ô thiếu thật. Các hàng đầy đủ có thể không đại diện cho toàn bộ dữ liệu thiếu.',
               '- CLR supervision của hàng train vốn thiếu sử dụng tâm log từ khởi tạo KNN; chỉ các tọa độ quan sát làm target. Đây là xấp xỉ vì CLR thật của hàng thiếu không xác định.',
               '- Target tự giám sát bị thay bằng median train trước khi tính CLR điều kiện, tránh rò rỉ target qua trung bình log.',
               '- Validation/test được che trước KNN. KNN và fallback chỉ dùng train. k=8 kế thừa, chưa tối ưu bằng inner CV.',
               '- T11 kiểm tra loss reducer không đọc nhãn ngoài target; không khẳng định conditioning không phụ thuộc giá trị khởi tạo.',
               '- Softmax đơn thuần không giữ được counts quan sát. Quy đổi về scale theo median log-ratio quan sát rồi khóa chính xác ô quan sát; output counts không buộc tổng bằng 1. CLR đánh giá được tính lại từ output cuối.',
               '- Zero fraction dự đoán dùng ngưỡng relative abundance <1e-4; không gọi softmax dương là structural zero. Shannon chỉ là chỉ số đa dạng biến thể.',
               '- Chưa thực hiện T18 tái lập DIABIMMUNE, T19 LTS, E1 ablation các khởi tạo, E2/drift nhiều vòng, downstream. Không kết luận về H1 hay ưu thế phylum CNN.',
               '- File X2 là kết quả điền toàn bộ dữ liệu từ artifact Stage A có sẵn; metrics chỉ lấy từ nhánh đánh giá độc lập, không từ file này.',
               '- Checkpoint chỉ train trên các địa điểm train, không refit toàn bộ dữ liệu sau đánh giá.',
               '', 'Các masks, dự đoán, lịch sử loss, C1/C2/C3, Shannon, tỷ lệ dưới ngưỡng và SHA256 đầu vào có trong results.json / audit_arrays.npz.']
    if improvement < 0:
        report[2:2] = ['**Kết quả âm: diffusion kém KNN trên pilot này. Không dùng file X2 làm dữ liệu đã được xác nhận chất lượng. Tests/mask/CLR đúng không chứng minh mô hình học tốt.**', '']
    Path('reports/stage_b_single_pass_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)
    print(f'DONE: {output}; {record["elapsed_seconds"]:.1f}s',flush=True)


if __name__ == '__main__':
    main()
