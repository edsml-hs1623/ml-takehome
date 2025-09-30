# Test transcription functionality - basic module import and function existence
from app.services.transcription import transcribe_audio

def test_transcribe_audio_function_exists():
    """Test that transcribe_audio function exists and is callable"""
    assert callable(transcribe_audio)

def test_transcribe_audio_import():
    """Test that transcription module can be imported"""
    from app.services import transcription
    assert hasattr(transcription, 'transcribe_audio')

