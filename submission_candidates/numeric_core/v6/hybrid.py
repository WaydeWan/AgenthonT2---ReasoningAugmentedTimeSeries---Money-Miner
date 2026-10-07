"""A fixed whole-path mixture of a short-window heavy-tail walk and V4.

Only the current unit's numeric history is fitted. The short-window branch is
inspired by the published M0 method, not an official-score reproduction.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
from .numeric import repair_covariance
from .sampler import sample_paths
from .targets import target_layout
from qfbench2_track_forecasting.targets import log_return_steps


def short_window_draws(unit, config, multipliers=None, df=7):
    unit=replace(unit,histories={a:s.loc[:pd.Timestamp(unit.asof)].sort_index() for a,s in unit.histories.items()})
    layout=target_layout(unit)
    columns,anchors={},[]
    for asset in unit.assets:
        history=unit.histories[asset].tail(300)
        gap=history.index.to_series().diff().dt.total_seconds()/86400
        maximum=max(5.,10*float(gap.dropna().median()))
        # The official target is cumulative log(1 + simple_return). The old
        # V6 short branch summed simple returns; preserve LEVEL parity while
        # correcting this branch for deployment to factor-return cards.
        step=(pd.Series(log_return_steps(history.to_numpy()), index=history.index)
              if unit.target_type=='log_return' else history.diff())
        columns[asset]=step.where(gap.notna() & (gap<=maximum))
        anchors.append(0. if unit.target_type=='log_return' else float(history.iloc[-1]))
    aligned=pd.DataFrame(columns).dropna()
    if len(aligned)<2:raise ValueError('Insufficient aligned short-window history')
    covariance=repair_covariance(np.atleast_2d(aligned.cov().to_numpy()),np.full(len(unit.assets),1e-10))
    steps=(layout.endpoints-layout.anchor_ticks[:,None]).ravel()
    if np.any(steps<1) or steps.max()>10000:raise ValueError('Invalid short-window path length')
    indexes=np.repeat(np.arange(len(unit.assets)),len(unit.horizons))
    mean=np.asarray(anchors)[indexes]+steps*aligned.mean().to_numpy()[indexes]
    shared=np.minimum(steps[:,None],steps[None,:])
    joint=shared*covariance[np.ix_(indexes,indexes)]
    multiplier=np.ones(len(unit.assets)) if multipliers is None else np.asarray(multipliers,dtype=float)
    if multiplier.shape!=(len(unit.assets),) or not np.isfinite(multiplier).all() or np.any(multiplier<=0):
        raise ValueError('Invalid short-window text multipliers')
    if np.any(multiplier!=1):
        times=np.arange(int(steps.max()))
        decay=2**(-times/config.text_decay_periods) if config.text_decay_periods>0 else np.ones(len(times))
        factors=np.exp(np.log(multiplier)[None,:]*decay[:,None])
        cumulative=np.cumsum(factors[:,:,None]*factors[:,None,:],axis=0)
        joint=covariance[np.ix_(indexes,indexes)]*cumulative[shared.astype(int)-1,indexes[:,None],indexes[None,:]]
    uq = config.drift_uq_strength if layout.frequency == 'daily' else 0.0
    if uq:
        joint += uq / len(aligned) * np.outer(steps, steps) * covariance[np.ix_(indexes, indexes)]
    joint=repair_covariance(joint,np.full(len(steps),1e-10))
    rng=np.random.default_rng(config.seed+301)
    noise=rng.standard_normal((config.draws,len(steps)))@np.linalg.cholesky(joint).T
    noise*=np.sqrt((df-2)/rng.chisquare(df,size=config.draws))[:,None]
    cube=(mean+noise).reshape(config.draws,len(unit.assets),len(unit.horizons))
    if not np.isfinite(cube).all():raise ValueError('Non-finite short-window forecast')
    return cube,{'drift_uq_strength_applied':uq,'history_limit':300,'aligned_steps':len(aligned),'student_df':df,
                 'native_anchors':anchors,'step_mean':aligned.mean().to_dict(),
                 'innovation_covariance':covariance.tolist(),
                 'method':'Native-level or return steps, joint covariance, covariance-standardized path Student-t'}


def sample_hybrid(unit,fit,config,multipliers=None):
    """Keep a complete path on each mixture branch; never average quantiles."""
    baseline=sample_paths(fit,config,multipliers)
    try:
        df=4 if config.numeric_variant=='M0_T4' else config.short_student_df
        candidate,detail=short_window_draws(unit,config,multipliers,df)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
        return baseline,{'variant':config.numeric_variant,'fallback':'V4','reason_type':type(exc).__name__}
    if config.numeric_variant=='M0_T4':
        output=candidate;weight=1.
    elif config.numeric_variant=='M0_T7_MIX75':
        output=baseline.copy()
        mask=np.arange(config.draws)%4!=0
        output[mask]=candidate[mask]
        weight=float(mask.mean())
    else:raise ValueError('Unknown hybrid runtime variant')
    return output,{'variant':config.numeric_variant,'short_window_weight':weight,
                   'fallback':None,'short_window':detail,'text_scale':'Shared stepwise decaying covariance adjustment'}
