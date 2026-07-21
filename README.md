# AMC under receiver hardware impairments

Investigating how hardware impairments of a low-cost SDR receiver 
(RTL-SDR, R828D tuner) degrade deep-learning modulation classifiers 
trained on synthetic data (RadioML 2018.01A), and whether calibrating 
the training-data impairment model to measured receiver characteristics 
closes the sim-to-real gap.

## Methodology

**Architecture:** ResNet (O'Shea et al., Table IV), fixed across all conditions —
not a variable under study. Structural fidelity to Table IV verified; exact
parameter-count match to the paper's stated figures not pursued (unresolved
ambiguity in the source, not load-bearing for a comparative design).

**Dataset:** RadioML 2018.01A (GOLD_XYZ_OSC.0001_1024.hdf5), full SNR range
(-20 to +30 dB), no truncation at data-generation stage.

**Experimental matrix (5 trained models, volume-matched, single fixed subset
reused across all):**
1. Baseline — clean synthetic data, no injected impairments.
2-4. Synthetic data with impairment parameters for RTL-SDR-class receivers,
   sourced from literature (device-specific for RTL-SDR Blog V4 / R828D where
   available, generic low-cost-SDR literature values as explicitly labeled
   fallback otherwise).
5. Synthetic data with impairment parameters calibrated from characteristics
   measured on the actual receiver (RTL-SDR Blog V4, R828D).

**Statistical protocol:** each condition trained over N independent runs
(train.seed varied, architecture/data fixed, N determined after pilot timing
run), reported as mean +/- std.

**Evaluation:**
- 5x5 synthetic-to-synthetic generalization matrix; in-domain (diagonal) and
  out-of-domain (off-diagonal mean) accuracy reported separately, plus
  generalization gap (ID - mean OOD).
- Held-out real OTA test (own receiver, ~7 modulations, ground truth via known
  frequency allocation/protocol per class) — single evaluation pass after all
  hyperparameters/datasets are frozen, reported as a separate primary result,
  not folded into the synthetic matrix average.
- Headline comparison metric: accuracy on SNR >= 0 dB subset (matches O'Shea's
  confusion-matrix reporting convention).
- Full accuracy-vs-SNR curve (2 dB bins) reported per model as the primary
  diagnostic artifact.

**Compute:** Kaggle free tier + rented Vast.ai GPU as needed; final run count
set after a pilot timing run on the smallest candidate subset size.

## Setup
...
## Reproduce
python src/train.py --config configs/baseline.yaml