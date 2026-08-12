#!/usr/bin/env python3
"""生成 openwakeword 校准/验证用的 TTS 音频（6 个官方唤醒词 + 负样本）。

校准数据越贴近真实唤醒词语音，分类器量化后精度越高：本仓库的峰值窗口验证显示，
仅用少量通用音频校准会让 hey_rhasspy / weather 分类器在真实触发窗口上坍缩
（得分 0.005 / 0.79）；加入 TTS 唤醒词正样本后提升到 0.97 / 1.00。

生成脚本只提交代码，不提交音频（在线 TTS 输出许可随服务商而定）。
需要网络访问 edge-tts（Microsoft 在线语音）。
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests/audio"
DEFAULT_VOICE = "en-US-ChristopherNeural"

CLIPS = {
    "alexa": ["Alexa", "Alexa, play some music"],
    "hey_jarvis": ["Hey Jarvis", "Hey Jarvis, what time is it"],
    "hey_mycroft": ["Hey Mycroft", "Hey Mycroft"],
    "hey_rhasspy": ["Hey Rhasspy", "Hey Rhasspy, start listening"],
    "weather": ["What's the weather", "What is the weather today"],
    "timer": ["Set a 10 minute timer", "Set a five minute timer"],
}
NEGATIVES = [
    "turn on the office lights",
    "open the door please",
    "good morning everyone",
]


async def synthesize(name: str, text: str, voice: str, output: Path) -> None:
    mp3 = output / f"{name}.mp3"
    await edge_tts.Communicate(text, voice).save(str(mp3))
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3), "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(output / f"{name}.wav"),
        ],
        check=True,
    )
    mp3.unlink(missing_ok=True)


async def main_async(output: Path, voice: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, texts in CLIPS.items():
        for index, text in enumerate(texts):
            await synthesize(f"{name}_{index}", text, voice, output)
    for index, text in enumerate(NEGATIVES):
        await synthesize(f"neg_{index}", text, voice, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    args = parser.parse_args()
    asyncio.run(main_async(args.output, args.voice))
    print(f"wrote calibration audio to {args.output}")


if __name__ == "__main__":
    main()
