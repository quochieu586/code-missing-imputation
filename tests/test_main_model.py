"""Milestone 1: actual initialization, time alignment and truth isolation, no torch."""
from dataclasses import replace
from pathlib import Path
import builtins
import numpy as np
import pytest

from longitudinal_impute.data import Dataset, load_csv, check_completed


def small_data(times=('3', '1', '2', '5')):
    counts = np.array([[20., 30.], [10., 20.], [30., 0.], [20., np.nan]])
    return Dataset(counts, np.full(4, 100.), ('a', 'b'), ('s1', 's1', 's1', 's2'),
                   ['a', 'b'], [{'a': str(a), 'b': str(b)} for a, b in counts], [2, 3, 4, 5],
                   times=times)


def test_loader_retains_raw_time_without_imposing_main_time_rules(tmp_path):
    source = tmp_path / 'data.csv'
    schema = tmp_path / 'schema.toml'
    source.write_text('id,time,a,b\ns,d1,10,20\n')
    schema.write_text('[dataset]\nentity="id"\ntime="time"\ntargets=["a","b"]\n')
    assert load_csv(source, schema).times == ('d1',)
    schema.write_text('[dataset]\nentity="id"\ntargets=["a","b"]\n')
    assert load_csv(source, schema).times is None


def test_pack_sort_padding_and_unpack_preserve_csv_order():
    from longitudinal_impute.main_model import pack_sequences, unpack_sequences
    data = small_data()
    values = np.arange(12).reshape(4, 3)
    packed, times, valid, indices = pack_sequences(data, values)
    assert packed.shape == (2, 3, 3)
    np.testing.assert_array_equal(indices, [[1, 2, 0], [3, -1, -1]])
    np.testing.assert_array_equal(times, [[0., 1., 2.], [4., 0., 0.]])
    np.testing.assert_array_equal(valid, indices >= 0)
    np.testing.assert_array_equal(packed[~valid], 0)
    np.testing.assert_array_equal(unpack_sequences(packed, indices), values)
    boolean, _, _, _ = pack_sequences(data, values > 3)
    assert boolean.dtype == bool


def test_dates_keep_real_gaps_and_shared_calendar_origin():
    from longitudinal_impute.main_model import pack_sequences
    data = small_data(('2020-01-10', '2020-01-01', '2020-01-03', '2020-01-05'))
    _, times, _, _ = pack_sequences(data, np.ones((4, 3)))
    np.testing.assert_array_equal(times, [[0., 2., 9.], [4., 0., 0.]])


@pytest.mark.parametrize('times', [None, ('1', '2'), ('1', '2020-01-01', '2', '3'),
                                  ('1', 'nan', '2', '3'), ('1', '1.0', '2', '3')])
def test_invalid_main_time_has_explicit_error(times):
    from longitudinal_impute.main_model import initialize
    with pytest.raises(ValueError, match='time'):
        initialize(small_data(times))


def test_initialization_matches_existing_tsagris_and_preserves_zero():
    from longitudinal_impute.main_model import initialize
    from longitudinal_impute.run import run_model
    from longitudinal_impute.baselines.tsagris import TsagrisKNN
    data = small_data()
    before = data.counts.copy()
    view = initialize(data)
    expected, status, _ = run_model(data, TsagrisKNN(), np.zeros_like(before, bool))
    assert status == 'completed'
    np.testing.assert_array_equal(view.counts, expected)
    np.testing.assert_array_equal(data.counts, before)
    assert view.counts[2, 1] == 0.
    assert not view.observed_mask[-1, -1]
    check_completed(data, view.counts, np.zeros_like(before, bool))


def test_evaluation_truth_is_absent_from_state_and_cannot_affect_features():
    from longitudinal_impute.main_model import initialize, clr_features
    data = small_data()
    hidden = np.zeros_like(data.counts, bool)
    hidden[0, 1] = True
    first = initialize(data, hidden)
    data.counts[hidden] = 70.
    data.rows[0]['b'] = '70'
    second = initialize(data, hidden)
    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(clr_features(first), clr_features(second))
    assert np.isnan(second.data.counts[hidden]).all()
    assert second.data.rows[0]['b'] == ''
    assert first.delta == second.delta


def test_training_view_excludes_q_before_initialization():
    from longitudinal_impute.main_model import prepare_views, clr_features
    data = small_data()
    # Leave a donor positive on both anchors after masking H and Q.
    data.counts[2, 1] = 10.
    data.rows[2]['b'] = '10'
    hidden = np.zeros_like(data.counts, bool)
    hidden[1, 0] = True
    q = np.zeros_like(hidden)
    q[0, 1] = True
    inference, training, actual_q, labels = prepare_views(data, hidden, train_mask=q)
    assert not (actual_q & hidden).any()
    assert labels[0, 1] == .3
    assert np.isnan(labels[~q]).all()
    assert np.isnan(training.data.counts[q]).all()
    changed = data.counts.copy()
    changed[q] = 70.
    _, again, _, changed_labels = prepare_views(replace(data, counts=changed), hidden, train_mask=q)
    np.testing.assert_array_equal(clr_features(training), clr_features(again))
    assert changed_labels[0, 1] == .7
    training.counts[0, 0] = 99.
    assert inference.counts[0, 0] == 20.
    bad_q = q | hidden
    with pytest.raises(ValueError, match='train mask'):
        prepare_views(data, hidden, train_mask=bad_q)


def test_seeded_q_is_repeatable_and_only_drawn_from_visible():
    from longitudinal_impute.main_model import prepare_views
    data = small_data()
    q1 = prepare_views(data, mask_rate=.1, seed=4)[2]
    q2 = prepare_views(data, mask_rate=.1, seed=4)[2]
    np.testing.assert_array_equal(q1, q2)
    assert not (q1 & ~np.isfinite(data.counts)).any()


def test_clr_is_finite_centered_and_does_not_change_counts():
    from longitudinal_impute.main_model import initialize, clr_features
    view = initialize(small_data())
    before = view.counts.copy()
    z = clr_features(view)
    assert z.shape == (4, 3) and np.isfinite(z).all()
    np.testing.assert_allclose(z.mean(axis=1), 0., atol=1e-14)
    np.testing.assert_array_equal(view.counts, before)


def test_no_donor_is_an_error_not_a_fallback():
    from longitudinal_impute.main_model import initialize
    data = small_data()
    counts = data.counts.copy()
    counts[:, 1] = np.nan
    with pytest.raises(ValueError, match='complete donor'):
        initialize(replace(data, counts=counts))


def test_zero_donor_anchor_is_reported_without_changing_mask():
    from longitudinal_impute.main_model import prepare_views
    data = small_data()
    hidden = np.zeros_like(data.counts, bool)
    hidden[1, 0] = True
    q = np.zeros_like(hidden)
    q[0, 1] = True
    # The only remaining complete donor has b=0, while row 1 is anchored on b.
    with pytest.raises(ValueError, match='zero mass'):
        prepare_views(data, hidden, train_mask=q)


@pytest.mark.parametrize('bad', [np.array([[0, 0]]), np.array([[0, 2]]),
                                np.array([[-2, 0]]), np.array([[0., 1.]])])
def test_unpack_rejects_invalid_mapping(bad):
    from longitudinal_impute.main_model import unpack_sequences
    with pytest.raises(ValueError, match='unpack'):
        unpack_sequences(np.zeros((1, 2, 3)), bad)


def test_initializer_zero_residual_stays_zero():
    from longitudinal_impute.main_model import initialize, clr_features
    data = small_data()
    data.counts[-1, 0] = 100.
    view = initialize(data)
    assert view.counts[-1, 1] == 0.
    assert np.isfinite(clr_features(view)).all()


def test_init_rejects_invalid_evaluation_masks():
    from longitudinal_impute.main_model import initialize
    data = small_data()
    for hidden in (np.zeros(data.counts.shape, int), np.zeros((1, 2), bool),
                   ~np.isfinite(data.counts)):
        with pytest.raises(ValueError, match='Evaluation mask'):
            initialize(data, hidden)


def test_packing_row_permutation_is_reversible():
    from longitudinal_impute.main_model import pack_sequences, unpack_sequences
    data = small_data()
    order = [3, 0, 2, 1]
    permuted = replace(data, counts=data.counts[order],
                       entities=tuple(data.entities[i] for i in order),
                       times=tuple(data.times[i] for i in order))
    values = np.arange(12).reshape(4, 3)
    first = pack_sequences(data, values)
    second = pack_sequences(permuted, values[order])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(unpack_sequences(second[0], second[3]), values[order])


def test_init_never_imports_torch(monkeypatch):
    from longitudinal_impute.main_model import initialize
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('Milestone 1 must not import torch')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert np.isfinite(initialize(small_data()).counts).all()


@pytest.mark.parametrize('name,schema', [
    ('covid', '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'),
    ('microbiome', '[dataset]\nentity="subject"\ntime="visit"\ntargets=["Bacteroides","Prevotella","Faecalibacterium","Akkermansia","Bifidobacterium"]\n'),
])
def test_two_dataset_schemas_use_same_initialization_and_packing(tmp_path, name, schema):
    from longitudinal_impute.main_model import initialize, clr_features, pack_sequences, unpack_sequences
    spec = tmp_path / 'schema.toml'
    spec.write_text(schema)
    data = load_csv(Path(__file__).parent / 'fixtures' / f'{name}_small.csv', spec)
    view = initialize(data)
    z = clr_features(view)
    packed, _, _, indices = pack_sequences(data, z)
    np.testing.assert_array_equal(unpack_sequences(packed, indices), z)
    assert z.shape[1] == len(data.targets) + 1
