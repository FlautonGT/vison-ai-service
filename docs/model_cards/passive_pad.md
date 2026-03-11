# Model Card: Passive PAD

## Intended use

- Image-based passive presentation attack detection for image-only API flows

## Evaluation expectations

- APCER, BPCER, ACER
- ROC and DET
- Attack-type breakdowns
- Cross-subject and session-disjoint evaluation

## Dataset cautions

- Keep attack types and sessions explicitly separated in evaluation
- Do not mix synthetic augmentation into evaluation unless labeled as such

## External conformance gap

- Internal metrics can align with ISO/IEC 30107 concepts, but they are not certification evidence
