"""Fixed scenario transforms extracted from the researched N1/N2/N3 formulas."""
import hashlib
import numpy as np


def rate_sources(base, anchors, seed):
    """Return S and L; a selected scenario gets one common scalar adjustment."""
    anchor_mean = float(np.mean(anchors))
    distance = anchor_mean + .25
    common = base.mean(axis=(1, 2))
    change = common - anchor_mean
    mask = np.zeros(len(base), dtype=bool)
    mask[np.random.default_rng(seed).permutation(len(base))[:round(.8 * len(base))]] = True
    chosen = mask & (change < 0.) & (distance > 0.)
    soft = base.copy()
    if chosen.any():
        with np.errstate(over='ignore', under='ignore', invalid='raise', divide='raise'):
            new_common = anchor_mean + distance * np.expm1(change[chosen] / distance)
        soft[chosen] = (base[chosen] - common[chosen, None, None]) + new_common[:, None, None]
    location = base.copy()
    shift = 0.
    if distance > 0.:
        target = float(np.median(soft.mean(axis=(1, 2))))
        shift = target - float(np.median(common))
        if shift != 0.:
            location += shift
        for _ in range(2):
            residual = target - float(np.median(location.mean(axis=(1, 2))))
            if residual == 0.:
                break
            location += residual
            shift += residual
    if not np.isfinite(soft).all() or not np.isfinite(location).all():
        raise ValueError('Nonfinite rate transform')
    return soft, location, {
        'mean_anchor': anchor_mean, 'soft_distance_pp': distance,
        'selected_downside_rows': int(chosen.sum()), 'soft_weight': .8,
        'location_scalar_shift': float(shift), 'active': distance > 0.,
        'actual_fit': False, 'new_innovations': False,
        'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
    }


def rate_combination(base, soft, location, candidate, seed):
    if candidate == 'N1':
        result = .5 * soft + .5 * location
        identical = np.all(soft == location, axis=(1, 2))
        result[identical] = soft[identical]
        detail = {'operation': 'same_row_paired_mean', 'soft_weight': .5}
    elif candidate == 'N2':
        payload = ('shape-rate-sl50-mixture-v1:' + str(seed)).encode()
        derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        mask = np.zeros(len(base), dtype=bool)
        mask[np.random.default_rng(derived).permutation(len(base))[:len(base) // 2]] = True
        result = location.copy()
        result[mask] = soft[mask]
        detail = {'operation': 'whole_joint_row_mixture', 'soft_rows': int(mask.sum()),
                  'location_rows': int((~mask).sum()), 'derived_seed': derived,
                  'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
    elif candidate == 'N3':
        result = location.copy()
        detail = {'operation': 'common_median_location_translation'}
    else:
        raise ValueError('Unknown numerical candidate')
    return result, {**detail, 'same_row_indices': True, 'new_innovations': False}


def transform_coordinates(base, transforms):
    out = base.copy()
    for ai, transform in enumerate(transforms):
        if transform == 'log_level':
            if np.any(out[:, ai] <= 0.):
                raise ValueError('Nonpositive FX or index scenario: log transform unavailable')
            out[:, ai] = np.log(out[:, ai])
    return out


def width_scale(base, transforms, alpha):
    values = transform_coordinates(base, transforms)
    center = values.mean(axis=0)
    changed = center[None] + alpha * (values - center[None])
    for ai, transform in enumerate(transforms):
        if transform == 'log_level':
            changed[:, ai] = np.exp(changed[:, ai])
    if not np.isfinite(changed).all():
        raise ValueError('Nonfinite width transform')
    return changed


def tail_sharpen(base, transforms, strength=.15):
    """Preserve tails in native output units, including exact tail sample bytes."""
    values = transform_coordinates(base, transforms)
    center = np.median(values, axis=0)
    ordered = np.sort(values, axis=0)
    low_rank = int(np.ceil(.05 * (len(base) - 1)))
    high_rank = int(np.floor(.95 * (len(base) - 1)))
    lower, upper = ordered[low_rank], ordered[high_rank]
    delta = values - center[None]
    width = np.where(delta < 0., center - lower, upper - center)
    interior = (width > 0.) & (np.abs(delta) < width)
    adjusted = values.copy()
    ratio = np.abs(delta[interior]) / width[interior]
    adjusted[interior] = np.broadcast_to(center, values.shape)[interior] + delta[interior] * (1. - strength * (1. - ratio ** 2) ** 2)
    out = base.copy()
    for ai, transform in enumerate(transforms):
        mask = interior[:, ai]
        out[:, ai][mask] = np.exp(adjusted[:, ai][mask]) if transform == 'log_level' else adjusted[:, ai][mask]
    return out, {'operation': 'tail_preserving_sharpen', 'strength': strength,
                 'low_rank': low_rank, 'high_rank': high_rank,
                 'changed_cells': int(np.count_nonzero(base != out)), 'actual_fit': False}
