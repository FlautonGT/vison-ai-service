# API Examples

## Capability discovery

```bash
curl http://127.0.0.1:8000/api/face/capabilities
```

## Quality assessment

```bash
curl -X POST http://127.0.0.1:8000/api/face/quality \
  -F "image=@sample.jpg"
```

Example response shape:

```json
{
  "face": {
    "detected": true,
    "confidence": 99.1
  },
  "quality": {
    "score": 82.4,
    "brightness": 55.2,
    "sharpness": 71.0
  },
  "attributes": {
    "mask": false,
    "hatCap": false,
    "majorOcclusion": false
  }
}
```

## Attribute assessment

```bash
curl -X POST http://127.0.0.1:8000/api/face/attributes \
  -F "image=@sample.jpg"
```

## Age and gender analysis with optional extras

```bash
curl -X POST http://127.0.0.1:8000/api/face/analyze \
  -F "image=@sample.jpg" \
  -F "includeAttributes=true" \
  -F "includeQuality=true"
```

## Verification similarity

```bash
curl -X POST http://127.0.0.1:8000/api/face/similarity \
  -F "image=@probe.jpg" \
  -F "embeddingStored=$(cat stored_embedding.json)"
```

## Passive PAD

```bash
curl -X POST http://127.0.0.1:8000/api/face/liveness \
  -F "image=@sample.jpg"
```

## Deepfake / manipulated-face detection

```bash
curl -X POST http://127.0.0.1:8000/api/face/deepfake \
  -F "image=@sample.jpg"
```
