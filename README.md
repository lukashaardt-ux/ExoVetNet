<!-- Banner -->

Signal-only dual-branch 1D-CNN for vetting Kepler exoplanet transits.
This project serves as a continuation of EclipseSieve, which determined the dependence that Exoplanet Vetters have on the catalog.

> *TL;DR* - ExoVetNet is a signal-only dual-branched 1D-CNN that vets Kepler transits from their light curves. ExoVetNet makes decisions independently of catalog-derived features. The model serves as a continuation of [EclipseSieve](https://github.com/lukashaardt-ux/EclipseSieve), which determined the immense catalog dependency of exoplanet vetting pipelines. ExoVetNet asks whether exoplanet vetting is still viable without catalog features. The model reached a 5-fold cross validation F1 score of 0.868 ± 0.01, which approaches EclipseSieve's catalog-driven F1 of 0.890. **Overall, ExoVetNet determined that signal-only vetting still faces catalog dependency.** This dependency is relocated from features to folding in preprocessing.

## Background

## Approach

## Data & preprocessing

## Model & training

## Results

## Null results

## The catalog-dependency audit

## Failure analysis

## Interpretability

## Candidate application

## Limitations

## Future work

## Reproducing

## References