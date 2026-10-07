"""Official forecast interface; numerical policy and explicit optional SEP."""
import argparse
import os
from pathlib import Path
import sys
from time import monotonic

from ..io import read_unit, required_draws, write_output
from .router import FinalOptions, forecast, DEFAULT_POLICY, array_hash
from .plugins import apply_text, TEXT_MODES


def run(panels, text, asof, output, *, policy=DEFAULT_POLICY, seed=0, adaptation_seconds=180., text_mode='off'):
    started = monotonic()
    if text_mode not in TEXT_MODES:
        raise ValueError('Unknown final text mode')
    unit = read_unit(Path(panels), asof)
    if required_draws(unit) > 20000:
        raise ValueError('Required joint draw count exceeds supported output')
    options = FinalOptions(policy=policy, draws=20000, seed=seed, adaptation_seconds=adaptation_seconds).validate()
    samples, diagnostics = forecast(unit, options)
    numeric_hash = array_hash(samples)
    samples, text_audit = apply_text(samples, unit=unit, textdir=Path(text), seed=seed, mode=text_mode)
    diagnostics.update(input=unit.input_diagnostics,
        text_mode=text_mode, text_used=bool(text_audit['applied']), text_source_attempted=text_mode == 'sep_v1', optional_plugin=text_audit,
        numeric_output_samples_sha256=numeric_hash, output_samples_sha256=array_hash(samples),
        official_text_argument=('Accepted for interface compatibility; not read by pure numerical policy' if text_mode == 'off'
                                else 'Only current-unit supplied corpus may be read by fixed production SEP adapter; no House'),
        elapsed_seconds_before_output_write=round(monotonic()-started, 6))
    if text_mode == 'sep_v1':
        diagnostics['method'] += ' with optional supplied-SEP scenario'
    write_output(unit, samples, Path(output), diagnostics)
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(prog='forecast', description=__doc__)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--asof', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        policy = os.environ.get('MONEY_MINER_FINAL_POLICY', DEFAULT_POLICY)
        seed = int(os.environ.get('QFBENCH_SEED', '0'))
        budget = float(os.environ.get('MONEY_MINER_ADAPTATION_SECONDS', '180'))
        text_mode = os.environ.get('MONEY_MINER_TEXT_MODE', 'off')
        run(args.panels, args.text, args.asof, args.out, policy=policy, seed=seed, adaptation_seconds=budget, text_mode=text_mode)
    except Exception as error:
        print('Forecast failed: '+type(error).__name__, file=sys.stderr)
        return 2
    print('Wrote numerical joint forecast and required sidecars.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
