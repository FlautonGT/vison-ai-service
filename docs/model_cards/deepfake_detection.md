# Model Card: Deepfake / Manipulated-Face Detection

## Intended use

- Internal manipulated-face and synthetic-face risk scoring for face verification workflows
- Not a standalone authenticity guarantee

## Inputs and outputs

- Input: aligned face crop or source image routed through shared face detection
- Output: attack score, thresholded decision, and risk level

## Evaluation expectations

- Confusion matrix
- ROC and DET curves
- Threshold search on validation data
- Slice reporting by available demographic proxy fields

## Dataset cautions

- Synthetic-only datasets are not sufficient for replay, print, or morph threats
- Kaggle mirrors do not override upstream dataset licenses

## External conformance gap

- No ISO/IEC 20059 or other external claim should be made without dedicated morphing-attack protocol testing
