# Model Card: Age, Gender, Attributes, and Quality

## Intended use

- Optional demographic and image-condition signals for e-KYC style flows
- Not identity truth or legal classification

## Outputs

- Age estimate
- Gender class
- Eyeglasses, sunglasses, mask, hat/cap, and major occlusion
- Brightness, illumination, blur/sharpness, and overall quality score

## Evaluation expectations

- Age MAE / RMSE
- Gender accuracy
- Multi-label F1 for attributes
- Regression error for quality
- Demographic slice reporting where lawful and available

## External conformance gap

- Quality alignment to ISO/IEC 29794-5 concepts is heuristic unless validated against a dedicated quality benchmark and protocol
