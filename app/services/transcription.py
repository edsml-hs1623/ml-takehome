import whisper
import tempfile
import shutil
from pathlib import Path
from fastapi import UploadFile


def transcribe_audio(audio_file: UploadFile, model_size="base", device="cpu") -> str:
    """
    Transcribe audio file using Whisper
    """
    model = whisper.load_model(model_size, device=device)

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        tmp_path = Path(tmp.name)

    result = model.transcribe(str(tmp_path))
    transcript = result["text"]

    return transcript
