"""Predeclared weak-intervention hypothesis, before any outcome evaluation."""

VERSION = 'text-scenario-v1'
SETTINGS = {
    'version': VERSION,
    'supported_assets': ['UST_2Y'],
    'supported_target_type': 'level',
    'supported_frequency': 'daily',
    'value_unit': 'percentage_points',
    'beta': .25,
    'mixture_mass': .10,
    'max_changed_fraction': .10,
    'minimum_ess_fraction': .98,
    'tenor_calendar_years': 2,
    'score': '-mean(min(abs((sample-view)/baseline_IQR),2))/2',
    'curve': 'piecewise linear from current target midpoint to dated SEP year-end medians',
    'median_priority': 'literal current median; otherwise exact supplied dot-count median',
    'missing_year': 'identity; no sparse-year interpolation',
    'longer_run': 'excluded; has no documented calendar convergence date',
    'extrapolation': 'forbidden',
    'calendar': 'pandas BDay endpoint (Monday-Friday); exact 2-calendar-year integration',
    'resampling': 'maximal-coupling proposals from uniform rows to weak mixture; constrained acceptance',
    'controls': ['beta0_identity', 'shuffled_row_scores'],
    'house_role': 'optional citation-checked allow/abstain veto; no direction or magnitude',
    'training': 'none; parameters are declared research priors, not learned or calibrated',
    'implemented_routes': ['whole_joint_row_reweight'],
    'unimplemented_routes': ['center_shift', 'width_change'],
    'accuracy_evaluation': False,
}

LIMITATIONS = [
    'Named hypothesis: the SEP path is used as a short-rate scenario under an expectations-hypothesis bridge with unchanged term premium.',
    'SEP participant assessments of appropriate policy are not market expectations or probabilities.',
    'A path of cross-sectional medians need not be the path of any individual participant.',
    'The current policy-band midpoint is a proxy for the future short rate; interpolation is an assumption, not a stated Fed commitment.',
    'The latest supplied current policy band is assumed still current at asof; omitted intervening decisions cannot be inferred.',
    'Current Treasury close may already incorporate the news; the scenario is not an estimated news surprise.',
    'Treasury term premia can change; this prototype holds their contribution to the yield change fixed.',
    'It is not fitted, calibrated, an identified causal effect, or a demonstrated accuracy improvement.',
    'Dates are cutoff-day resolved, not intraday; publication-versus-market-close ordering is not established.',
    'Constraints on empirical ESS may reject resampling proposals, so the final empirical distribution is not claimed to equal the theoretical mixture exactly.',
]
