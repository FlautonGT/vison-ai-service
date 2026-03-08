import asyncio, time, pathlib
import cv2
import numpy as np
from app.core.models import ModelRegistry
from app.api import face_router

async def main():
    reg = ModelRegistry()
    await reg.load_all()

    img_path = next(pathlib.Path(r"C:/Users/USER/Website Pribadi/Vison/vison-ai-service/benchmark_data").glob("ai_faces/*.jpg"))
    img1 = cv2.imread(str(img_path))
    img2 = np.zeros((1200, 1200, 3), dtype=np.uint8)
    img3 = np.ones((200, 200, 3), dtype=np.uint8) * 255

    for label, img, kwargs in [
        ("valid", img1, {}),
        ("blank_no_fallback", img2, {}),
        ("small_face_candidate", img3, {"allow_precropped_fallback": True}),
    ]:
        t0 = time.perf_counter()
        face, proc, err = face_router._detect_and_validate(img, reg, False, False, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        print(label, "err", bool(err), "elapsed_ms", round(dt, 2), "face", (face is not None), "proc", (proc is not None), "has_err_payload", getattr(err, 'status_code', None))

asyncio.run(main())
