# Vendored research pipeline

`predict_success.py` in this folder is an unmodified copy of the training
pipeline from [haddadyousef/NBA-Draft-prediction](https://github.com/haddadyousef/NBA-Draft-prediction),
commit `b0db46844007a939bfc9d644f613b8815e6f10a4` (2026-01-06), the source of
truth for the Shopping by Skill research (MIT Sloan SSAC 2026, Basketball
Track, paper ID 171).

It is kept here for reference and citation only — `scripts/export_artifacts.py`
does not import it. It re-implements the same feature engineering (per-40
stats, the engineered shooting/rebounding/playmaking/defense ratios, the
conference one-hot encoding) and the same model family (Random Forest,
HistGradientBoosting, XGBoost, soft VotingClassifier over stratified 5-fold
CV with SMOTE inside each fold), but changes two things deliberately — see
`README.md`'s model card section for why:

1. **A real 2018 draft-class holdout.** The published pipeline has no
   season-based holdout (`college_stats_final.csv` has no `season` column),
   so the 2018 draft class is inside its training set. `export_artifacts.py`
   excludes those named players from training so the 2018 case study is a
   genuine forecast, not a replay.
2. **Full out-of-fold coverage.** The published pipeline's
   `sti_probabilities.csv` only keeps the single best-accuracy fold per
   model (overwritten by the other 4), covering roughly a fifth of players.
   `export_artifacts.py` aggregates held-out predictions across all 5 folds
   so every training-set player gets one real out-of-fold prediction.

It also fits the per-skill K-means/PCA clustering once on the training
population and persists it (`artifacts/preprocess.joblib`), instead of
refitting per fold and discarding it — matching what a server needs to score
a player consistently.

`LICENSE` of the source repository is included alongside for attribution.
