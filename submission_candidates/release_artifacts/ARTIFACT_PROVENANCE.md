# Money Miner T2: runtime code and development provenance

Release prepared on 2026-10-07 for Development, Team 379. The final image digest and release-config identity determine the submission. This record describes sources and limitations; it is not organizer approval or an accuracy certificate.

## Contents and model disclosure

The image contains ordinary non-neural numerical estimation code, source parsers, and fixed algorithm settings. It contains no offline fitted coefficients, trained weights, embeddings, retrieval index, historical prediction/answer cache, research results, public task corpus, development credentials, or language-model weights. `models.json` is empty because no fitted model artifact or API model is bundled or used. Parameters are estimated afresh from the current input unit at runtime and discarded after that unit. This follows the artifact policy's allowance for ordinary numerical forecasting code, rather than inventing a fictitious pretrained model name or global training cutoff for per-unit estimates. The policy's wording about runtime estimates remains subject to organizer interpretation; descriptor validation cannot certify eligibility.

There are no House calls and no network fetches during forecasting. Earlier NVIDIA development diagnostics are research only and their responses are not packaged.

## Source versions and licenses

The submitted code is an MIT-licensed derivative of the team's V6/V8 forecasting work (`agenthon_t2_rase_v1 contributors`, copyright 2026) with subsequent numerical adaptation and text-parsing code. The original MIT notice is preserved in `/app/LICENSE`. Original teammate V8 source revision was `bc59490`; this release does not claim unchanged V8 behavior. The source allowlist records every exact file SHA256, and `/app/release-identity.json` binds its manifest and configuration. The public repository commit and image digest provide release identity; source code has its actual 2026 development date, not a fabricated historical publication date.

Official dependencies: Track 2 public source `60509df4ad0756443f4af8dc9500aed99a995694`; toolkit `bd01548e34d21fd660d88fd06157078fb25ece4e` (2.6.0), with the bundled license/notice files. Statistical runtime: Python 3.13.12, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, PyArrow 23.0.0, scikit-learn 1.6.1, threadpoolctl 3.6.0, jsonschema 4.26.0, PyYAML 6.0.3. The Dockerfile pins direct versions and the Python base digest. Transitive build inputs are not all artifact-hash pinned; only the actually published image digest is immutable.

## Per-unit numerical estimation and cutoff

Inputs are only the official mounted card, forecast specification, panels, and optional documents. The official cutoff loader and production bounding code exclude observations/versions unavailable at the card's as-of. If a daily input has no `available_at` column, its observation date is used as the supplied-data proxy; this is not proof that the data reconstructs first-release historical vintages. No external data is downloaded at evaluation time.

The V6 base combines a recent-history numerical branch and a conditional path branch to produce 20,000 complete joint scenarios in native target units. Each row preserves all assets and horizons. Adaptive policies are restricted to supported daily UST level grids. They select among V6, paired S/L, whole-row S/L, location, and central-tail proposals using at most 24 mature month-end origins from this input. Training labels must be available before the first confirmation origin; the confirmation segment uses only labels matured by the outer cutoff. Training and confirmation are separated and unconfirmed changes revert to V6. `shrink` uses the learned interpolation coefficient; `conservative` caps it at 0.5. Fitting has a cooperative 180-second budget. Unsupported/failed adaptation falls back to the same raw V6 distribution and reports the reason. This is not a hard interruption guarantee for a stalled operating-system call.

The fixed policy uses the existing N3 location transformation on eligible multi-cell UST grids and inherited tail transformation on other eligible grids. In the exact daily singleton log-level FX branch, all final candidates retain the raw V6 samples, with no inherited 0.85 width multiplier and no width-fitting artifact. This choice was made for transfer robustness, not because every currency or date preferred raw width.

No learned state transfers across units. Each output rationale identifies the executed policy, estimation decision, fallback, and current-unit cutoff. There is no single truthful offline training-cutoff date for runtime-only estimates, and none is invented here.

## Optional policy-path text component

Only the `policy-text` configuration enables `sep_v1`. It reads dated documents from the current unit using the production corpus loader. It requires explicit current policy-rate bounds and sufficient annual median SEP projections. It applies only to the supported UST 2-year level forecast and does not extrapolate missing years or use the long-run dot as a dated observation.

The component converts the literal policy path into a weak numerical scenario under an explicitly unverified constant-term-premium assumption. It lightly reweights complete joint sample rows; it does not replace individual assets independently. Its fixed limits preserve effective sample size and bound changed rows. Missing, conflicting, out-of-date, unsupported, or malformed evidence leaves the numerical samples unchanged. No LLM decides direction, model, magnitude, or probability. Source availability bounds are distinguished from exact original publication timestamps.

## Development, selection, and calibration data

Architecture and constants were developed using public historical numerical data and official public example/practice inputs. Development examined multiple periods, including 2010–2023 numerical evaluations, later historical windows, cross-currency/cross-tenor transfers, and public 2015/2019/2023/2024 policy documents. Research used final-vintage historical proxies where historical first-release vintages were unavailable. Those research datasets, results, target answers, model outputs, and fitted artifacts are not included in the image.

The S/L proposal constants (including the 0.8 row mask, anchor-plus-0.25 location, and 0.5 combination), the inherited 0.15 tail transformation, proposal family, runtime window sizes, and confirmation rule have development-history provenance. Writing these choices as code and fitting within the current unit does not erase their development/selection sources. The official policy applies cutoff restrictions to fitting, adaptation, selection, and calibration. We do not claim that disclosure alone resolves every interpretation of historically developed algorithm choices. The evaluator receives no stored unit-answer lookup or future-data artifact.

The most recent 42-case, same-sample numerical comparison was a post-hoc development comparison, not an independent hidden test. Adaptive shrink and conservative showed modest mean improvements over raw V6; this is not proof of statistical significance, global optimality, or leaderboard performance. The policy-path experiment had 14 eligible cases but only two actual interventions, both on singleton UST2Y grids where the inherited N1/N3 rule is the same. These limited observations do not establish broad text efficacy. A changed numerical base cannot inherit another base's text-accuracy result.

The release is a set of three Development candidates, not a claim of a perfect predictor. No official submission or Final attempt is performed by building or packaging it.
