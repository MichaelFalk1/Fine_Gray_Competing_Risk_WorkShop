"""Test the canonical notebook directly; no duplicate implementation."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'matplotlib'))
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_schema_and_syntax():
    import nbformat
    nb = nbformat.read(ROOT / 'workshop.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, f'cell {cell.id}', 'exec')


@pytest.fixture(scope='session')
def ws():
    nb = json.loads((ROOT / 'workshop.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and set(cell['metadata'].get('tags', [])) & {
            'imports', 'definitions', 'verification'}:
            source = cell['source']
            exec(''.join(source) if isinstance(source, list) else source, namespace)
    return namespace


def test_simulation_and_truth(ws):
    a, truth = ws['simulate_cohort']()
    b, _ = ws['simulate_cohort']()
    pd.testing.assert_frame_equal(a, b)
    c, _ = ws['simulate_cohort'](seed=999)
    assert not a.equals(c)
    assert len(ws['FEATURES']) == 8
    assert set(a.event) == {0, 1, 2}
    assert np.isfinite(a.time).all() and (a.time > 0).all()
    assert a.subject_id.is_unique
    assert set(a.columns) == {'subject_id', 'time', 'event', *ws['FEATURES']}
    t = np.linspace(0, 20, 50)
    f, g = ws['true_cif'](truth, t), ws['true_cif'](truth, t, 2)
    assert (f >= 0).all() and (f <= 1).all()
    assert (np.diff(f, axis=1) >= 0).all() and (f+g <= 1+1e-12).all()


def test_risk_sets_and_censoring(ws):
    d = pd.DataFrame({'subject_id': range(5), 'time': [1., 2., 3., 4., 5.], 'event': [2, 0, 1, 1, 2]})
    table = ws['risk_table'](d, 4.)
    assert table['Cox included'].tolist() == [False, False, False, True, True]
    assert table['FG member'].tolist() == [True, False, False, True, True]
    np.testing.assert_allclose(table['FG contribution'], [.75, 0, 0, 1, 1])
    assert .72/.9 == pytest.approx(.8)
    d['event'] = [2, 2, 1, 1, 2]
    np.testing.assert_allclose(ws['risk_weights'](d, 4.), [1, 1, 0, 1, 1])
    # Independent check against lifelines' reverse-KM step values.
    from lifelines import KaplanMeierFitter
    d['event'] = [2, 0, 1, 1, 0]
    km = KaplanMeierFitter().fit(d.time, event_observed=d.event == 0)
    queries = np.array([1., 2., 2.5, 4., 5.])
    np.testing.assert_allclose(ws['censor_survival_left'](d, queries), km.predict(queries-1e-9))


@pytest.mark.parametrize('m', [0, 1, 4, 1000])
def test_ncc_eligibility_and_measurement(ws, m):
    df, _ = ws['simulate_cohort'](80)
    out = df[['subject_id', 'time', 'event']]
    sample = ws['sample_ncc'](out, m)
    expected = set(out.loc[out.event == 1, 'subject_id'])
    for row in sample['records']:
        chosen = row['chosen']
        assert row['case_id'] not in chosen
        assert len(chosen) == len(set(chosen)) == min(m, len(row['eligible']))
        assert set(chosen) <= set(row['eligible'])
        assert (out.set_index('subject_id').loc[chosen, 'time'] >= row['time']).all()
        expected.update(chosen)
    assert expected == sample['measured']
    visible = ws['measured_view'](df, expected)
    assert visible.loc[~visible.subject_id.isin(expected), ws['FEATURES']].isna().all().all()
    assert visible.loc[visible.subject_id.isin(expected), ws['FEATURES']].notna().all().all()
    if m == 1000:
        assert sample['control_occurrences'] > len(sample['controls'])  # repeated persons


@pytest.mark.parametrize('q', [0., .1, .25, 1.])
def test_case_cohort_fixed_overlap(ws, q):
    d, _ = ws['simulate_cohort'](80)
    out = d[['subject_id', 'time', 'event']]
    s = ws['sample_case_cohort'](out, q)
    assert s == ws['sample_case_cohort'](out, q)
    assert len(s['baseline']) == int(np.floor(q*len(d)))
    assert s['cases'] <= s['measured']
    assert len(s['measured']) == len(s['baseline'])+len(s['added'])
    if q == 1:
        assert s['measured'] == set(d.subject_id)
    if q == 0:
        assert not s['baseline'] and s['measured'] == s['cases']


def test_full_risk_analytical_and_shift_invariance(ws):
    assert ws['verify_full_risk_math']()
    d, _ = ws['simulate_cohort'](80)
    cases, t, lw = ws['fg_design'](d)
    eta = np.random.default_rng(1).normal(size=len(d))
    l, g = ws['fg_loss_score'](eta, cases, lw)
    l2, g2 = ws['fg_loss_score'](eta+10000, cases, lw)
    np.testing.assert_allclose(l, l2, atol=1e-10)
    np.testing.assert_allclose(g, g2, atol=1e-10)
    assert np.isfinite(l) and np.isfinite(g).all()
    assert abs(g.sum()) < 1e-12


def test_invalid_and_sparse_inputs(ws):
    d, _ = ws['simulate_cohort'](10)
    with pytest.raises(ValueError): ws['sample_case_cohort'](d, -.1)
    with pytest.raises(ValueError): ws['sample_ncc'](d, -1)
    d.event = 0
    assert not ws['sample_ncc'](d)['records']
    with pytest.raises(ValueError, match='No strokes'): ws['fg_design'](d)
    d.loc[:1, 'event'] = 1
    d.loc[:1, 'time'] = 1.
    with pytest.raises(ValueError, match='distinct'): ws['fg_design'](d)


def test_external_pycmprsk_reference(ws):
    """Optional independent reference, not required by participants."""
    ref = pytest.importorskip('pycmprsk')
    d, _ = ws['simulate_cohort'](160, seed=401)
    x = d[ws['FEATURES']].to_numpy()
    fit = ws['fit_linear_fg'](d, x)
    external = ref.crr(d.time.to_numpy(), d.event.to_numpy(), cov1=x, gtol=1e-9, maxiter=100)
    # Access verified against the package's public result fields.
    np.testing.assert_allclose(fit['coef'], external.coef, atol=2e-5)
    np.testing.assert_allclose(fit['times'], external.uftime, atol=1e-12)
    np.testing.assert_allclose(fit['jumps'], external.bfitj, atol=2e-6)
    prediction = ref.predict_crr(external, cov1=x[:3])
    ours = ws['predict_fg'](fit['times'], fit['baseline'], x[:3]@fit['coef'], fit['times'])
    np.testing.assert_allclose(ours.T, prediction[:, 1:], atol=2e-5)


def test_optional_tensorflow(ws):
    tf = pytest.importorskip('tensorflow')
    assert ws['verify_tensorflow'](tf)
    tf.keras.utils.set_random_seed(8)
    model = ws['build_model'](tf)
    assert model.count_params() == 361
    assert model(np.zeros((4, 8))).shape == (4, 1)
    d, _ = ws['simulate_cohort'](50)
    c, _, lw = ws['fg_design'](d)
    with tf.GradientTape() as tape:
        loss = ws['tf_fg_loss'](tf, model(d[ws['FEATURES']].to_numpy()), c, tf.constant(lw))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)


def test_prototype_full_coverage_and_inclusion(ws):
    d = pd.DataFrame({'subject_id': range(5), 'time': [1., 2., 3., 4., 5.],
                      'event': [1, 2, 1, 2, 0]})
    cases, _, lw = ws['fg_design'](d)
    for mode, m, q in [('CC', 2, 1.), ('NCC', 100, .25)]:
        design = ws['prototype_design'](d, mode, m, q)
        np.testing.assert_array_equal(design['ids'], d.subject_id)
        np.testing.assert_array_equal(design['cases'], cases)
        np.testing.assert_allclose(design['logw'], lw)
        assert design['zero_support'] == 0
    ncc = ws['prototype_design'](d, 'NCC', m=1)
    # Last two noncases eligible in draws of size 4 and 2.
    np.testing.assert_allclose(ncc['pi'], [1., .25, 1., .625, .625])
    cc = ws['prototype_design'](d, 'CC', q=.4)
    np.testing.assert_allclose(cc['pi'], [1., .4, 1., .4, .4])
    eta = np.linspace(-.2, .2, len(ncc['ids']))
    _, gradient = ws['fg_loss_score'](eta, ncc['cases'], ncc['logw'])
    eps = 1e-6
    numerical = [(ws['fg_loss_score'](eta+eps*e, ncc['cases'], ncc['logw'])[0]
                  -ws['fg_loss_score'](eta-eps*e, ncc['cases'], ncc['logw'])[0])/(2*eps)
                 for e in np.eye(len(eta))]
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_prototype_zero_support_and_cases_only(ws):
    d = pd.DataFrame({'subject_id': range(4), 'time': [1., 2., 3., 4.], 'event': [2, 1, 0, 1]})
    design = ws['prototype_design'](d, 'NCC', m=100)
    assert design['zero_support'] == 1
    assert design['pi'][0] == 0
    assert 0 not in design['ids']
    for mode, m, q in [('NCC', 0, .25), ('CC', 2, 0.), ('CC', 2, .01)]:
        with pytest.raises(ValueError, match='cases-only'):
            ws['prototype_design'](d, mode, m=m, q=q)


def test_optional_prototype_training_and_hidden_features(ws):
    tf = pytest.importorskip('tensorflow')
    tf.keras.utils.set_random_seed(8)
    initial = ws['build_model'](tf).get_weights()
    d, _ = ws['simulate_cohort'](60)
    holdout, _ = ws['simulate_cohort'](10, seed=98)
    design = ws['prototype_design'](d, 'CC', q=.4)
    measured = ws['measured_view'](d, set(design['ids']))
    a = ws['fit_sampled_prototype'](tf, measured, design, holdout[ws['FEATURES']], initial, 2, .01)
    # Unmeasured feature values cannot affect training or preprocessing.
    measured.loc[~measured.subject_id.isin(design['ids']), ws['FEATURES']] = 1e99
    b = ws['fit_sampled_prototype'](tf, measured, design, holdout[ws['FEATURES']], initial, 2, .01)
    np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-10)
    assert a['model'].count_params() == 361
    assert np.isfinite(a['history']).all()
    full = ws['prototype_design'](d, 'CC', q=1.)
    complete = ws['fit_sampled_prototype'](tf, d, full, holdout[ws['FEATURES']], initial, 2, .01)
    # Independently reproduce the full-risk loop to check q=1 recovery.
    scaler = ws['StandardScaler']().fit(d[ws['FEATURES']])
    x = tf.constant(scaler.transform(d[ws['FEATURES']]), dtype=tf.float64)
    model = ws['build_model'](tf); model.set_weights(initial)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
    cases, _, lw = ws['fg_design'](d)
    for _ in range(2):
        with tf.GradientTape() as tape:
            loss = ws['tf_fg_loss'](tf, model(x), cases, tf.constant(lw))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
    expected = model(scaler.transform(holdout[ws['FEATURES']])).numpy().ravel()
    np.testing.assert_allclose(complete['scores'], expected, atol=1e-10)


def test_prototype_cif_hand_calculation_and_shift(ws):
    # Four uncensored people: stroke, death, stroke, death; all measured.
    d = pd.DataFrame({'subject_id': range(4), 'time': [1., 2., 3., 4.], 'event': [1, 2, 1, 2]})
    design = ws['prototype_design'](d, 'CC', q=1.)
    query = np.array([0., 1., 2., 3., 4.])
    training = np.log([1., 1., 2., 2.])
    prediction = np.log([1., 2.])
    # Denominators 6 at first stroke, 5 at second (prior death retained).
    baseline = np.array([0., 1/6, 1/6, 1/6+1/5, 1/6+1/5])
    expected = -np.expm1(-np.array([1., 2.])[:, None]*baseline)
    actual = ws['prototype_cif'](design, training, prediction, query)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    shifted = ws['prototype_cif'](design, training+10000, prediction+10000, query)
    np.testing.assert_allclose(shifted, actual, atol=1e-11)
    # An illustrative two-person sample with inclusion weights 1 and 1/0.5.
    toy = {'times': np.array([1.]), 'logw': np.log([[1., 2.]])}
    np.testing.assert_allclose(ws['prototype_cif'](toy, [0., 0.], [0.], [0., 1.]),
                               [[0., -np.expm1(-1/3)]])


def test_prototype_cif_full_coverage_matches_reference_and_bounds(ws):
    d, _ = ws['simulate_cohort'](80)
    x = d[ws['FEATURES']].to_numpy()
    fit = ws['fit_linear_fg'](d, x)
    query = np.linspace(0, d.time.max(), 40)
    design = ws['prototype_design'](d, 'CC', q=1.)
    actual = ws['prototype_cif'](design, x@fit['coef'], x[:4]@fit['coef'], query)
    expected = ws['predict_fg'](fit['times'], fit['baseline'], x[:4]@fit['coef'], query)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    for mode in ['NCC', 'CC']:
        selected = ws['prototype_design'](d, mode, m=2, q=.25)
        # Fixed numerical scores test the prediction algebra, not a trained DNN.
        scores = np.linspace(-1, 1, len(selected['ids']))
        values = ws['prototype_cif'](selected, scores, [-2., 0., 2.], query)
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        assert (np.diff(values, axis=1) >= -1e-12).all()
        assert (values[:, 0] == 0).all()
    with pytest.raises(ValueError, match='finite'):
        ws['prototype_cif'](design, np.full(len(d), np.nan), [0.], query)
