import os
import re
from typing import List

import ctc_segmentation
import numpy as np
import pandas as pd
import pysrt
import torch
import torchaudio
from nltk.tokenize import sent_tokenize
from rixvox.api import get_audio_metadata, get_media_file
from rixvox.text import normalize_text
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor


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

    probs = probs[0].cpu().numpy() if probs.ndim == 3 else probs.cpu().numpy()
    # Get nr of encoded CTC frames in the encoder without padding.
    # I.e. the number of "tokens" the audio was encoded into.
    ctc_frames = calculate_w2v_output_length(audio_frames, chunk_size=30)

    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / ctc_frames / samplerate
    print(f"Index duration: {config.index_duration}")
    print(f"audio_frames: {audio_frames}")
    print(f"ctc_frames: {ctc_frames}")
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
    speech_nr = row["anf_nummer"]

    # Extract the audio from the start to the end of the speech with ffmpeg
    os.makedirs("data/audio/speeches", exist_ok=True)
    audiofile = row["downloadurl"]
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


def get_probs(speech_metadata, pad=False):
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
        # If audio chunk is shorter than 30 seconds, pad it to 30 seconds
        if audio_chunk.shape[1] < chunk_size * sr:
            padding = torch.zeros((1, chunk_size * sr - audio_chunk.shape[1]))
            audio_chunk = torch.cat([audio_chunk, padding], dim=1)
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


def is_only_non_alphanumeric(text):
    """
    re.match returns a match object if the pattern is found and None otherwise.
    """
    # Contains only 1 or more non-alphanumeric characters
    return re.match(r"^[^a-zA-Z0-9]+$", text) is not None


def word_tokenize(text):
    text = row["anf_text"].split(" ")  # word tokenization
    text = [token for token in text if is_only_non_alphanumeric(token) is False]
    return text


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


def calculate_w2v_output_length(
    audio_frames: int,
    chunk_size: int,
    conv_stride: list[int] = [5, 2, 2, 2, 2, 2, 2],
    sample_rate: int = 16000,
    frames_first_logit: int = 400,
):
    """
    Calculate the number of output characters from the wav2vec2 model based
    on the chunking strategy and the number of audio frames.

    The wav2vec2-large model outputs one logit per 320 audio frames. The exception
    is the first logit, which is output after 400 audio frames (the model's minimum
    input length).

    We need to take into account the first logit, otherwise the alignment will slowly
    drift over time for long audio files when chunking the audio for batched inference.

    Parameters
    ----------
    audio_frames
        Number of audio frames in the audio file, or part of audio file to be aligned.
    chunk_size
        Number of seconds to chunk the audio by for batched inference.
    conv_stride
        The convolutional stride of the wav2vec2 model (see model.config.conv_stride).
        The product sum of the list is the number of audio frames per output logit.
        Defaults to the conv_stride of wav2vec2-large.
    sample_rate
        The sample rate of the w2v processor, default 16000.
    frames_first_logit
        First logit consists of more frames than the rest. Wav2vec2-large outputs
        the first logit after 400 frames.
    """
    frames_per_logit = np.prod(conv_stride)
    extra_frames = frames_first_logit - frames_per_logit

    frames_per_full_chunk = chunk_size * sample_rate
    n_full_chunks = audio_frames // frames_per_full_chunk

    # Calculate the number of logit outputs for the full size chunks
    logits_per_full_chunk = (frames_per_full_chunk - extra_frames) // frames_per_logit
    n_full_chunk_logits = n_full_chunks * logits_per_full_chunk

    # Calculate the number of logit outputs for the last chunk (may be shorter than the chunk size)
    n_last_chunk_frames = audio_frames % frames_per_full_chunk

    if n_last_chunk_frames == 0:
        n_last_chunk_logits = 0
    elif n_last_chunk_frames < frames_first_logit:
        # We'll pad the last chunk to 400 frames if it's shorter than the model's minimum input length
        n_last_chunk_logits = 1
    else:
        n_last_chunk_logits = (n_last_chunk_frames - extra_frames) // frames_per_logit

    return n_full_chunk_logits + n_last_chunk_logits


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
    get_media_file(meta["downloadurl"][0], progress_bar=True)

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
        probs, audio_length = get_probs(speech_metadata, pad=True)
        align_probs.append(probs)
        audio_frames.append(audio_length)

    normalized_transcripts = []
    original_transcripts = []
    for i, row in meta.iterrows():
        # Chunk text according to what granularity we want alignment timestamps.
        # We sentence tokenize here, but we could also word tokenize, and then
        # at a later stage decide how to create subtitle chunks from word timestamps.

        transcript = sent_tokenize(row["anf_text"])
        # transcript = word_tokenize(row["anf_text"])
        normalized_transcript = [normalize_text(token).upper() for token in transcript]
        normalized_transcripts.append(normalized_transcript)
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
            segment["start"] += float(speech_metadata["start_speech"])
            segment["end"] += float(speech_metadata["start_speech"])
        alignments.append(align)

    # Flatten the alignments
    alignments = [segment for speech in alignments for segment in speech]
    # Flatten the original transcripts
    transcripts = [token for speech in original_transcripts for token in speech]

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
    basename = os.path.basename(meta["downloadurl"][0]).replace(".mp4", ".srt")
    subtitle_filename = os.path.join("data/audio", basename)
    subs.save(subtitle_filename)

    # Embed the subtitles into the video container as a subtitle track
    output_video = os.path.join("data/audio", basename.replace(".srt", "_subtitled.mp4"))
    original_video = os.path.join("data/audio", basename.replace(".srt", ".mp4"))
    os.system(
        f"ffmpeg -i {original_video} -i {subtitle_filename} -c copy -c:s mov_text -metadata:s:s:0 language=swe {output_video}"
    )
