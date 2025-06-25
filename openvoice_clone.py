import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import srt
from faster_whisper import WhisperModel

class OpenVoiceInfer:
    def __init__(self, device="cpu", language="en"):
        self.device = device
        self.language = language
        self.CHECKPOINTS_PATH = "checkpoints_v2"

        self.converter = ToneColorConverter(f"{self.CHECKPOINTS_PATH}/converter/config.json", device=device)
        self.converter.load_ckpt(f"{self.CHECKPOINTS_PATH}/converter/checkpoint.pth")

        self.tts = TTS(language=language, device=device)
        self.speaker_ids = self.tts.hps.data.spk2id
        print("[👥] Available speakers:", self.speaker_ids)
        self.base_speaker_key = list(self.speaker_ids.keys())[0]
        self.base_speaker_id = self.speaker_ids[self.base_speaker_key]

        se_extractor.model = WhisperModel("large", device="cpu", compute_type="float32")

    def infer(self, text: str, reference_audio_path: str, output_audio_path: str, segments_dir: str):
        print(f"[🗣️] '{text}' | Ref: {reference_audio_path} → Out: {output_audio_path}")

        target_se, _ = se_extractor.get_se(reference_audio_path, self.converter, target_dir=segments_dir, vad=False)
        tmp_base_path = "./tmp/base.wav"
        os.makedirs("tmp", exist_ok=True)

        # Generate base voice with default speaker
        self.tts.tts_to_file(text, self.base_speaker_id, tmp_base_path)

        # Apply tone color
        self.converter.convert(
            audio_src_path=tmp_base_path,
            src_se=target_se,
            tgt_se=target_se,
            output_path=output_audio_path,
            message="@MyShell"
        )
        print(f"[✅] Done: {output_audio_path}")

if __name__ == "__main__":
    SRT_PATH = "/Users/prateek/work/github/video-project/data/translated_polished.srt"
    SOURCE_SEGMENTS_DIR = "/Users/prateek/work/github/video-project/data/segments"
    DEST_SEGMENTS_DIR = "/Users/prateek/work/github/video-project/data/processed"
    os.makedirs(DEST_SEGMENTS_DIR, exist_ok=True)

    with open(SRT_PATH, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    infer = OpenVoiceInfer(device="cpu", language="EN")

    for i, sub in enumerate(subtitles):
        print(f"\n[🎞️] Processing subtitle {i}")
        ref_audio = os.path.join(SOURCE_SEGMENTS_DIR, f"ref_{i:04d}.wav")
        output_audio = os.path.join(DEST_SEGMENTS_DIR, f"converted_{i:04d}.wav")
        infer.infer(sub.content, ref_audio, output_audio, DEST_SEGMENTS_DIR)