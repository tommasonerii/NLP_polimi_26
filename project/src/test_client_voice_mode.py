"""Smoke test for the PoliMillionaire client speech mode.

The updated client supports speech games through:

- `client.game.start(competition_id, mode="speech")`
- `game.fetch_audio_question()`
- `game.fetch_audio_option_next()`
- `game.fetch_audio_option(index)` for replaying already delivered options

This script starts a speech-mode game, downloads the question and option WAV
files, saves them locally, and optionally plays them. It does not submit an
answer unless `--answer-letter` is provided.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave


DEFAULT_API_URL = "http://131.175.15.22:51111/"


def add_client_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    client_parent = repo_root / "api_client" / "NLP_assignment_api_client"
    if str(client_parent) not in sys.path:
        sys.path.insert(0, str(client_parent))


def env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def save_audio(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def inspect_wav(path: Path) -> str:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            duration = frames / float(rate) if rate else 0.0
            return f"{duration:.2f}s, {rate} Hz, {channels} ch, {width * 8}-bit"
    except wave.Error as exc:
        return f"not a valid WAV according to wave module: {exc}"


def play_audio(path: Path) -> bool:
    if sys.platform.startswith("win"):
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return True
        except Exception:
            return False

    for player in ("afplay", "paplay", "aplay"):
        executable = shutil.which(player)
        if executable:
            try:
                subprocess.run([executable, str(path)], check=True)
                return True
            except subprocess.SubprocessError:
                return False
    return False


def option_id_from_letter(game, letter: str) -> int:
    index = ord(letter.upper()) - ord("A")
    question = game.current_question
    if question is None:
        raise RuntimeError("No current question available.")
    if index < 0 or index >= len(question.options):
        raise ValueError(f"Invalid answer letter {letter!r}; expected A-D.")
    return question.options[index].id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the official client speech mode.")
    parser.add_argument(
        "--api-url",
        default=env_value("POLIMILLIONAIRE_API_URL", "API_URL") or DEFAULT_API_URL,
        help=f"API base URL. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--username",
        default=env_value("POLIMILLIONAIRE_USERNAME", "USERNAME"),
        help="Login username. Can also use POLIMILLIONAIRE_USERNAME or USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=env_value("POLIMILLIONAIRE_PASSWORD", "PASSWORD"),
        help="Login password. Can also use POLIMILLIONAIRE_PASSWORD or PASSWORD.",
    )
    parser.add_argument("--competition-id", type=int, default=0, help="Competition ID. Default: 0.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/voice_mode",
        help="Directory where WAV files are saved. Default: artifacts/voice_mode.",
    )
    parser.add_argument(
        "--options",
        type=int,
        default=4,
        help="Number of option audios to fetch sequentially. Default: 4.",
    )
    parser.add_argument(
        "--test-replay",
        action="store_true",
        help="After fetching options sequentially, also test fetch_audio_option(index) replay.",
    )
    parser.add_argument("--play", action="store_true", help="Play each downloaded WAV locally.")
    parser.add_argument(
        "--answer-letter",
        choices=["A", "B", "C", "D", "a", "b", "c", "d"],
        help="Optional: submit an answer after fetching audio. Omit for no-answer smoke test.",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Also fetch the speech-mode leaderboard for the competition.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.username or not args.password:
        print(
            "Missing credentials. Set POLIMILLIONAIRE_USERNAME/POLIMILLIONAIRE_PASSWORD "
            "or pass --username and --password.",
            file=sys.stderr,
        )
        return 2

    if args.options < 0 or args.options > 4:
        print("--options must be between 0 and 4.", file=sys.stderr)
        return 2

    add_client_to_path()
    from millionaire_client import MillionaireClient

    output_dir = Path(args.output_dir)
    client = MillionaireClient(args.api_url)
    user = client.login(args.username, args.password)
    print(f"Logged in as {user.username}.")

    game = client.game.start(args.competition_id, mode="speech")
    print(f"Session ID: {game.session_id}")
    print(f"Mode: {game.mode}")
    print(f"Level: {game.current_level}")

    question_path = output_dir / f"session_{game.session_id}_level_{game.current_level}_question.wav"
    print("Fetching question audio...")
    save_audio(game.fetch_audio_question(), question_path)
    print(f"Saved question: {question_path} ({inspect_wav(question_path)})")
    if args.play:
        play_audio(question_path)

    option_paths: list[Path] = []
    for i in range(args.options):
        letter = chr(ord("A") + i)
        option_path = output_dir / f"session_{game.session_id}_level_{game.current_level}_option_{letter}.wav"
        print(f"Fetching option {letter} audio with fetch_audio_option_next()...")
        save_audio(game.fetch_audio_option_next(), option_path)
        option_paths.append(option_path)
        print(f"Saved option {letter}: {option_path} ({inspect_wav(option_path)})")
        if args.play:
            play_audio(option_path)

    if args.test_replay:
        for i, original_path in enumerate(option_paths):
            letter = chr(ord("A") + i)
            replay_path = output_dir / f"session_{game.session_id}_level_{game.current_level}_option_{letter}_replay.wav"
            print(f"Replaying option {letter} with fetch_audio_option({i})...")
            save_audio(game.fetch_audio_option(i), replay_path)
            same = replay_path.read_bytes() == original_path.read_bytes()
            print(f"Saved replay {letter}: {replay_path} ({inspect_wav(replay_path)}), same_bytes={same}")

    game.refresh_state()
    if game.time_remaining is not None:
        print(f"Time remaining after audio fetch: {game.time_remaining:.1f}s")
    print("No answer submitted by default.")

    if args.answer_letter:
        option_id = option_id_from_letter(game, args.answer_letter)
        print(f"Submitting answer {args.answer_letter.upper()} -> option_id={option_id}")
        result = game.answer(option_id)
        print(
            f"Result: correct={result.correct}, timed_out={result.timed_out}, "
            f"game_over={result.game_over}, earned=${result.earned_amount:,.2f}"
        )

    if args.leaderboard:
        leaderboard = client.leaderboard.get(args.competition_id, limit=10, mode="speech")
        print(f"Speech leaderboard: {leaderboard.competition.name}")
        for rank, entry in enumerate(leaderboard.entries, start=1):
            print(f"{rank}. {entry.username}: ${entry.score:,.2f} (level {entry.reached_level})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
