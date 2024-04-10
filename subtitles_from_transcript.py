import os
from typing import List

import ctc_segmentation
import numpy as np
import pysrt
import torch
import torchaudio
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.api import get_audio_metadata, get_media_file
from rixvox.text import normalize_text


def align_with_transcript(
    transcripts: List[str],
    probs: torch.Tensor,
    audio_frames: int,
    processor: Wav2Vec2Processor,
    samplerate: int = 16000,
):
    # Tokenize transcripts
    vocab = processor.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = processor.tokenizer(transcript.replace("\n", " ").upper())["input_ids"]
        tok_ids = np.array(tok_ids, dtype="int")
        tokens.append(tok_ids[tok_ids != unk_id])

    # Get nr of characters in the model output (if batched, it's the second dimension)
    probs = probs[0].cpu().numpy() if probs.ndim == 3 else probs.cpu().numpy()
    probs_size = probs.shape[1] if probs.ndim == 3 else probs.shape[0]

    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / probs_size / samplerate
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs, ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, transcripts
    )
    return [
        {"text": t, "start": p[0], "end": p[1], "conf": p[2]}
        for t, p in zip(transcripts, segments)
    ]


def split_speech_from_media(row):
    start_speech = row["start"]
    end_speech = row["end"]
    speech_nr = row["number"]

    # Extract the audio from the start to the end of the speech with ffmpeg
    os.makedirs("data/audio/speeches", exist_ok=True)
    audiofile = row["downloadfileurl"]
    audiofile = os.path.join("data/audio", audiofile.rsplit("/")[-1])
    basename = os.path.basename(audiofile)
    speech_audiofile = os.path.join("data/audio/speeches", f"{basename}_{speech_nr}.wav")

    # Convert the video to wav 16kHz mono from the start to the end of the speech
    os.system(
        f"ffmpeg -i {audiofile} -ac 1 -ar 16000 -ss {start_speech} -to {end_speech} {speech_audiofile}"
    )

    return {
        "speech_audiofile": speech_audiofile,
        "start_speech": start_speech,
        "end_speech": end_speech,
    }


def get_probs(speech_metadata):
    # Load the audio file
    speech_audiofile = speech_metadata["speech_audiofile"]
    audio_input, sr = torchaudio.load(speech_audiofile)
    audio_input.to(device).half()  # Convert to half precision

    # Split the audio into chunks of 30 seconds
    chunk_size = 30
    audio_chunks = torch.split(audio_input, chunk_size * sr, dim=1)

    # Transcribe each audio chunk
    all_probs = []

    for audio_chunk in audio_chunks:
        input_values = (
            processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding="longest")
            .input_values.to(device)
            .squeeze(dim=0)
        )
        with torch.inference_mode():
            logits = model(input_values.half()).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        all_probs.append(probs)

    # Concatenate the probabilities
    align_probs = torch.cat(all_probs, dim=1)
    return align_probs, len(audio_input[0])


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss,ms" format
    expected by pysrt.
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
    )

    # Get metadata for a riksdag debate through the API for a given debate document id
    meta = get_audio_metadata(rel_dok_id="hb10625")
    get_media_file(meta["downloadfileurl"][0], progress_bar=True)

    meta["end"] = meta["start"] + meta["duration"]

    speeches_metadata = []
    for i, row in meta.iterrows():
        # Create a wav file with only the speech's audio
        audio_speech = split_speech_from_media(row)
        speeches_metadata.append(audio_speech)

    align_probs = []
    audio_frames = []
    for speech_metadata in tqdm(speeches_metadata):
        # Run transcription but only keep the probabilities for alignment
        probs, audio_length = get_probs(speech_metadata)
        align_probs.append(probs)
        audio_frames.append(audio_length)

    normalized_transcripts = []
    original_transcripts = []
    for i, row in meta.iterrows():
        # Chunk text according to what granularity we want alignment timestamps.
        # We sentence tokenize here, but we could also word tokenize, and then
        # at a later stage decide how to create subtitle chunks from word timestamps.
        transcript = sent_tokenize(row["anftext"])
        normalized_transcripts.append(
            [normalize_text(sentence).upper() for sentence in transcript]
        )
        original_transcripts.append(transcript)

    alignments = []
    for i, speech_metadata in enumerate(tqdm(speeches_metadata)):
        # Alignment of audio with transcript
        align = align_with_transcript(
            normalized_transcripts[i],
            align_probs[i],
            audio_frames[i],
            processor,
        )
        for segment in align:
            segment["start"] += speech_metadata["start_speech"]
            segment["end"] += speech_metadata["start_speech"]
        alignments.append(align)

    # Flatten the alignments
    alignments = [segment for speech in alignments for segment in speech]
    # Flatten the original transcripts
    transcripts = [sentence for speech in original_transcripts for sentence in speech]

    # Create a subtitles file from the timestamps
    subs = pysrt.SubRipFile()

    for i, alignment in enumerate(alignments):
        # pysrt expects "hh:mm:ss,ms" format where ms is 3 digits
        # We currently have seconds with decimals
        start = format_timestamp(alignment["start"])
        end = format_timestamp(alignment["end"])
        text = transcripts[i]
        subs.append(pysrt.SubRipItem(index=i, start=start, end=end, text=text))

    # Save the subtitles file
    basename = os.path.basename(meta["downloadfileurl"][0]).replace(".mp4", ".srt")
    subtitle_filename = os.path.join("data/audio", basename)
    subs.save(subtitle_filename)

    # Embed the subtitles into the video container as a subtitle track
    output_video = os.path.join("data/audio", basename.replace(".srt", "_subtitled.mp4"))
    original_video = os.path.join("data/audio", basename.replace(".srt", ".mp4"))
    os.system(
        f"ffmpeg -i {original_video} -i {subtitle_filename} -c copy -c:s mov_text -metadata:s:s:0 language=swe {output_video}"
    )
