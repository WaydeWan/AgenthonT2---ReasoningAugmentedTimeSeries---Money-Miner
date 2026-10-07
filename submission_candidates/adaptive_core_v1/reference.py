"""Recover exact frozen daily-level M0 moments without sampling predictions.

These helpers intentionally accept no labels. General official monthly and
log-return cards require a separate target-aware adapter; historical datasets
in this migration are explicitly daily native-level proxy cases.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def construct(assets, horizons, mu, sigma, anchors, *, factorization=None):
    assets, horizons = list(assets), list(horizons)
    if (not assets or len(set(assets)) != len(assets) or not horizons
            or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or h < 1 for h in horizons)):
        raise ValueError('Invalid declared daily-level grid')
    mu, sigma, anchors = map(lambda v: np.asarray(v, dtype=float), (mu, sigma, anchors))
    a = len(assets)
    if (mu.shape != (a,) or sigma.shape != (a, a) or anchors.shape != (a,)
            or not all(np.isfinite(v).all() for v in (mu, sigma, anchors))
            or not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-14)):
        raise ValueError('Invalid asset step moments')
    ai = np.repeat(np.arange(a), len(horizons))
    h = np.tile(horizons, a)
    mean = anchors[ai]+h*mu[ai]
    covariance = np.minimum.outer(h, h)*sigma[np.ix_(ai, ai)]
    covariance = covariance+np.eye(len(h))*1e-10
    covariance = covariance+np.eye(len(h))*1e-9
    used = 'full_covariance'
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        covariance = np.diag(np.diag(covariance))
        np.linalg.cholesky(covariance)
        used = 'documented_diagonal_fallback'
    if factorization is not None and used != factorization:
        raise ValueError('Saved factorization does not match reconstructed covariance')
    return {'mean': mean, 'covariance': covariance, 'factorization': used,
            'assets': assets, 'horizons': horizons, 'cell_order': 'asset_major_horizon_minor',
            'mean_sha256': array_hash(mean), 'covariance_sha256': array_hash(covariance),
            'target_type': 'level', 'target_frequency': 'daily', 'outcome_used': False}


def from_diagnostics(diag, assets, horizons, asof):
    assets, horizons = list(assets), list(horizons)
    if (diag.get('status') != 'ok' or diag.get('actual_model') != 'public_method_local_M0_daily_level'
            or diag.get('current_truth_used') is not False or diag.get('asof') != asof
            or diag.get('output_assets') != assets or diag.get('output_horizons') != horizons
            or diag.get('generation_assets') != sorted(assets)
            or diag.get('generation_horizons') != sorted(horizons)
            or diag.get('diagonal_jitter_additions') != [1e-10, 1e-9]):
        raise ValueError('Saved public-method reference identity mismatch')
    sorted_assets = diag['generation_assets']
    mu = np.asarray(diag['mean_step_sorted_assets'])
    sigma = np.asarray(diag['covariance_sorted_assets'])
    ix = [sorted_assets.index(a) for a in assets]
    for asset in assets:
        if diag['per_asset'][asset]['last_observation'] != asof:
            raise ValueError('Historical proxy requires exact cutoff anchor')
    # Reconstruct in ORIGINAL generation order first: Cholesky fallback is a
    # numerical decision and must not be reevaluated under a new permutation.
    generated = construct(sorted_assets, diag['generation_horizons'], mu, sigma,
                          [diag['per_asset'][a]['anchor'] for a in sorted_assets],
                          factorization=diag['factorization'])
    old_cells = [(a, h) for a in sorted_assets for h in diag['generation_horizons']]
    order = [old_cells.index((a, h)) for a in assets for h in horizons]
    mean = generated['mean'][order]
    covariance = generated['covariance'][np.ix_(order, order)]
    return {**generated, 'assets': assets, 'horizons': horizons, 'mean': mean,
            'covariance': covariance, 'mean_sha256': array_hash(mean),
            'covariance_sha256': array_hash(covariance),
            'input_history_sha256': diag['history_sha256'],
            'source': 'saved_public_method_exact_moments'}


def from_histories(histories, assets, horizons, asof, *, expected_history_hash=None):
    """Cut off independently, then last300 differences/gap-filter/date-align."""
    assets, horizons = list(assets), list(horizons)
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize() or pd.isna(cutoff):
        raise ValueError('Invalid cutoff')
    bounded = {}
    for asset in assets:
        source = histories[asset]
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError('Dated independent history required')
        selected = source.loc[source.index <= cutoff].copy().sort_index().astype(float)
        if (selected.empty or not selected.index.is_unique or selected.index[-1] != cutoff
                or not np.isfinite(selected.to_numpy()).all()):
            raise ValueError('Invalid cutoff history')
        bounded[asset] = selected
    identity = hashlib.sha256(json.dumps({'assets': assets, 'horizons': horizons,
                                         'asof': str(cutoff.date())}, sort_keys=True).encode())
    for asset in assets:
        identity.update(asset.encode())
        identity.update(np.asarray(bounded[asset].index.asi8, dtype='<i8').tobytes())
        identity.update(np.asarray(bounded[asset].to_numpy(), dtype='<f8').tobytes())
    history_hash = identity.hexdigest()
    if expected_history_hash is not None and history_hash != expected_history_hash:
        raise ValueError('Cutoff history differs from original forecast input')
    columns = {}
    for asset in sorted(assets):
        trailing = bounded[asset].tail(300)
        if len(trailing) < 3:
            raise ValueError('Insufficient trailing history')
        gaps = trailing.index.to_series().diff().dt.days
        good = gaps.notna() & (gaps <= max(5., 10.*float(gaps.dropna().median())))
        columns[asset] = trailing.diff().loc[good]
    aligned = pd.concat(columns, axis=1, join='inner').dropna().sort_index()
    if len(aligned) < 2 or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError('Insufficient aligned increments')
    matrix = aligned.to_numpy(dtype=float)
    ga, gh = sorted(assets), sorted(horizons)
    generated = construct(ga, gh, matrix.mean(axis=0), np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1)),
                          [bounded[a].iloc[-1] for a in ga])
    cells = [(a, h) for a in ga for h in gh]
    order = [cells.index((a, h)) for a in assets for h in horizons]
    mean, cov = generated['mean'][order], generated['covariance'][np.ix_(order, order)]
    return {**generated, 'assets': assets, 'horizons': horizons, 'mean': mean,
            'covariance': cov, 'mean_sha256': array_hash(mean), 'covariance_sha256': array_hash(cov),
            'input_history_sha256': history_hash, 'aligned_increment_rows': len(aligned),
            'source': 'frozen_cutoff_history_exact_moments'}
