## Align transcriptions with audio

Examples of how to align manually transcribed text with audio and create subtitles.

### Requirements

Create a virtual environment and install the required packages:

```bash
python3 -m venv venvs/align
source venvs/align/bin/activate
pip install -r requirements.txt
```

> [!IMPORTANT]
> `ffmpeg` is a system dependency required to run the scripts. Install it with `sudo apt install ffmpeg`.

### Usage

See `subtitles_from_transcript.py` for how to create subtitles from speech protocols.

```bash
python subtitles_from_transcript.py
```

The resulting subtitles and media files are saved in `data/audio`.