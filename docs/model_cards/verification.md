# Model Card: Face Verification

## Intended use

- 1:1 face verification using embeddings and calibrated thresholds

## Training approach

- Embedding backbone plus ArcMargin classification head
- Subject-disjoint train, validation, and pair manifests

## Evaluation expectations

- ROC and DET
- EER
- FMR/FNMR operating points
- Threshold calibration for deployment

## External conformance gap

- Internal reporting can align with ISO/IEC 19795 and FIDO-style metrics, but field deployment still needs scenario-specific validation
