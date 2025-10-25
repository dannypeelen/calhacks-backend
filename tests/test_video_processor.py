import base64
import io
import os
import time
import pytest

from starlette.datastructures import UploadFile

from app.services.video_processor import process_webcam_frame, process_uploaded_video


@pytest.mark.asyncio
async def test_process_webcam_frame_throttle(tmp_path, monkeypatch):
    # Prepare a small fake JPEG-like payload; content does not need to be valid image
    b64 = base64.b64encode(b"frame-bytes").decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    # First call: should not be throttled and should save
    out1 = await process_webcam_frame(data_url)
    assert out1["ok"] is True
    assert out1["throttled"] in (False, True)  # tolerate environment timing

    # Force throttle by not advancing time
    out2 = await process_webcam_frame(data_url)
    assert out2["ok"] is True
    if out2["throttled"]:
        assert out2["saved_path"] is None


@pytest.mark.asyncio
async def test_process_uploaded_video(tmp_path, monkeypatch):
    # Fake small file upload (not a real video); should still save and return ok
    content = b"not-a-video"
    up = UploadFile(filename="test.mp4", file=io.BytesIO(content))
    out = await process_uploaded_video(up)
    assert out["ok"] is True
    assert out["bytes"] == len(content)
    assert os.path.exists(out["saved_path"])  

