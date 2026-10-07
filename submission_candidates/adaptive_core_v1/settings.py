"""Engineering choices frozen before any accuracy evaluation of this module."""
from copy import deepcopy

MODES = ('conservative', 'mixture', 'shrink')
FAMILY = ('V6', 'paired_SL', 'row_mixture_SL', 'location_L', 'central_tail')
INNER_DRAWS = 512
MIN_PREFIX = 756
MAX_ORIGINS = 24
MIN_TRAIN = 8
MIN_CONFIRM = 4
MAX_CELLS = 24
TAILS = (.01, .05, .95, .99)
CAP = 8.

SETTINGS = {
    'method': 'current_unit_purged_fit_then_single_confirmation_v1',
    'scope': 'Daily native level/yield, all UST identifiers, 1 through 24 target cells only; singleton uses official5/7,0,2/7 weights',
    'unsupported': 'Unmodified production V6; no .85 width or .15 tail postprocessing',
    'family': list(FAMILY), 'modes': list(MODES),
    'inner_draws': INNER_DRAWS, 'maximum_origins': MAX_ORIGINS,
    'minimum_prefix_observations_per_asset': MIN_PREFIX,
    'minimum_training_origins_after_purge': MIN_TRAIN,
    'minimum_confirmation_origins': MIN_CONFIRM,
    'origins': 'Last at most24 exact business month ends with all exact horizon endpoints mature at outer cutoff',
    'confirmation': 'Last max(4,floor(n/4)) origins. All fitting uses earlier origins with all endpoints and actual selected label vintages strictly before first confirmation origin.',
    'training_labels': 'Latest vintage available before first confirmation origin, never latest outer vintage if it was not yet published',
    'confirmation_labels': 'Latest version available at outer cutoff; features always origin-vintage snapshots',
    'fitter': 'Select lowest mean-loss proposal on training only, then one bounded scalar beta in[0,1] toward V6; include exact beta0/1; no validation refit',
    'solver': {'method': 'bounded', 'maxiter': 24, 'xatol': .01},
    'conservative': 'Same selected proposal, beta=min(fitted_beta,.5)',
    'shrink': 'Same selected proposal, beta=fitted_beta',
    'mixture': 'Whole joint rows; proposal weights proportional to positive training improvement relative to V6; total proposal mass=min(.5,sum_relative_gain/(1+sum_relative_gain))',
    'continue_gate': 'For each already-fitted policy, confirmation mean clipped loss below V6 AND wins>losses; otherwise return V6. Not a significance test.',
    'reference': 'Analytic Gaussian M0 expected errors, full min(h1,h2)*covariance and separate1e-10/1e-9 jitters; ordered-pair variogram p=.5; zero denominator->1',
    'objective': 'Official raw fairCRPS/variogram/tail pinball, expected-error normalization, equal-origin clipped[0,8] loss',
    'failure': 'Any failed historical origin invalidates adaptation; no dropping failed origins. Return unmodified V6 with explicit reason.',
    'vintage_boundary': 'Production raw-current-unit adapter; absent available_at uses zero-day daily-level proxy, explicitly disclosed. Not a first-release-vintage reconstruction.',
    'research_provenance': {
        'V6': 'Previously researched model architecture and hyperparameters, not invented or selected solely at each historical cutoff.',
        'SL': 'Reuse researched rate_sources with .8 row mask and anchor+.25 distance; paired and whole-row alternatives use .5 structural S/L split.',
        'central_tail': 'Full strength1 proposal, not old selected strength.15; the transformation family itself originated in prior development.',
        'new_regularization': '24 origins, 8/4 minimum split, half-mass/cap, one-dimensional solver and confirmation gate are fixed engineering choices, not results of this module accuracy evaluation.',
        'boundary': 'Cutoff-local fitting does not erase development-data selection of the architecture, candidate family or constants. Does not prove competition policy compliance or unbiased historical performance.'},
    'global_optimality_claimed': False, 'statistical_independence_claimed': False,
    'accuracy_validated': False, 'official_commit': '60509df4ad0756443f4af8dc9500aed99a995694',
}


def settings():
    return deepcopy(SETTINGS)
