# PoliMillionaire API README

This file explains how to use the Python client provided in the `NLP_assignment_api_client.zip` package to interact with the "Who wants to be a PoliMillionaire?" game.

## Where the client is

The client (extracted from the official `NLP_assignment_api_client.zip` provided in the tutorials) is versioned in the repo at:

```text
api_client/NLP_assignment_api_client/
|-- PoliMillionaire.ipynb        # official API tutorial
`-- millionaire_client/
    |-- __init__.py
    |-- client.py
    |-- auth.py
    |-- game.py
    |-- competitions.py
    |-- leaderboard.py
    |-- models.py
    |-- base.py
    `-- exceptions.py
```

Locally it is enough to add the parent to `sys.path`:

```python
import sys
sys.path.append("api_client/NLP_assignment_api_client")

from millionaire_client import MillionaireClient, AuthenticationError
```

In the final project, on Colab, the `millionaire_client` folder must be in the same parent directory as the notebook, or that parent directory must be added to `sys.path`.

Recommended example on Google Drive:

```text
MyDrive/
`-- Colab Notebooks/
    `-- NLP_assignment/
        |-- PoliMillionaire.ipynb
        `-- millionaire_client/
```

## Setup in Colab

```python
from google.colab import drive
drive.mount("/content/gdrive/")
```

```python
import sys

package_parent_dir = "/content/gdrive/MyDrive/Colab Notebooks/NLP_assignment"

if package_parent_dir not in sys.path:
    sys.path.append(package_parent_dir)
```

```python
from millionaire_client import MillionaireClient, AuthenticationError
```

## Server URL

The assignment indicates this endpoint:

```python
API_URL = "http://131.175.15.22:51111/"
```

Note: the assignment says the site may not be accessible from the PoliMi Wi-Fi network due to a block on the port.

## Registration and login

First you must register from the browser on the site:

```text
http://131.175.15.22:51111/
```

Then in the notebook you use the login:

```python
client = MillionaireClient(API_URL)

try:
    user = client.login(username, password)
    print(f"Welcome, {user.username}! Role: {user.role}")
except AuthenticationError as e:
    print(f"Login failed: {e}")
```

Method used internally:

```text
POST /api/auth/login
```

The client automatically saves the authentication cookie in the `requests` session.

## Main object: MillionaireClient

`MillionaireClient` exposes four modules:

```python
client.auth          # authentication
client.competitions  # available competitions
client.game          # games and answers
client.leaderboard   # leaderboards
```

It also has handy shortcuts:

```python
client.login(username, password)
client.logout()
client.user
client.is_authenticated
```

The `game` module supports two modes:

```text
text    # default: text question and options
speech  # question and options as WAV audio downloaded from the server
```

## Competitions

To see the available competitions:

```python
competitions = client.competitions.list_all()

for comp in competitions:
    print(comp.id, comp.name, comp.max_levels)
```

API method:

```text
GET /api/competitions
```

To get the detailed configuration:

```python
config = client.competitions.get_config(competition_id=1)
print(config.name)
print(config.max_levels)
print(config.money_pyramid)
```

API method:

```text
GET /api/competitions/{competition_id}/config
```

## Starting a game

```python
comp_id = 1
game = client.game.start(competition_id=comp_id)

print(game.session_id)
print(game.current_level)
print(game.earned_amount)
```

API method:

```text
POST /api/game/start
```

The result is a `GameSession` object.

To start a game in voice mode:

```python
game = client.game.start(competition_id=comp_id, mode="speech")

print(game.session_id)
print(game.mode)  # "speech"
```

In the request body the client also sends `mode`:

```text
POST /api/game/start
{"competitionId": 1, "mode": "speech"}
```

## Reading the current question

```python
question = game.current_question

print(question.id)
print(question.level)
print(question.text)

for opt in question.options:
    print(opt.id, opt.text)
```

Main fields:

```text
question.id       # question id
question.text     # question text
question.level    # level
question.options  # list of Option
```

Each `Option` has:

```text
opt.id
opt.text
```

## Available time

Each question has a timeout, indicated in the assignment as a maximum of 30 seconds.

```python
time_left = game.time_remaining
print(time_left)
```

If you answer too late, the server can return a timeout even if the chosen option was correct.

In `speech` mode, the 30-second timer starts after requesting the audio of the last option. The correct flow is therefore: question audio, options A-D audio, state refresh, answer.

## Speech/audio mode

The voice mode is not a stream and does not use MP3. The client downloads complete audio files over HTTP and returns raw `bytes`. The client docstrings indicate WAV, so it is best to save them with the `.wav` extension.

Endpoints used by the client:

```text
GET /api/game/{session_id}/audio/question
GET /api/game/{session_id}/audio/option/next
GET /api/game/{session_id}/audio/option/{index}
```

Python methods:

```python
question_audio = game.fetch_audio_question()
option_a_audio = game.fetch_audio_option_next()
option_b_audio = game.fetch_audio_option_next()
option_c_audio = game.fetch_audio_option_next()
option_d_audio = game.fetch_audio_option_next()
```

The options must be requested in sequence with `fetch_audio_option_next()`: first A, then B, C, and D. After an option has been delivered, it can be replayed with `fetch_audio_option(index)`, where `index` is `0` for A, `1` for B, `2` for C, `3` for D.

Minimal example:

```python
from pathlib import Path

game = client.game.start(competition_id=comp_id, mode="speech")

out_dir = Path("artifacts/voice_mode")
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "question.wav").write_bytes(game.fetch_audio_question())

option_map = {}
for i in range(4):
    letter = chr(65 + i)  # A, B, C, D
    (out_dir / f"option_{letter}.wav").write_bytes(game.fetch_audio_option_next())
    option_map[letter] = game.current_question.options[i].id

game.refresh_state()
print("Time remaining:", game.time_remaining)

answer_letter = "A"
result = game.answer(option_map[answer_letter])
```

Local smoke test:

```powershell
C:\ProgramData\miniconda3\python.exe project/src/test_client_voice_mode.py --competition-id 0 --options 4 --test-replay --play --leaderboard
```

The script saves the WAVs in `artifacts/voice_mode/`. By default it does not submit answers; to submit an answer add `--answer-letter A`.

## Answering

Answer via option id:

```python
result = game.answer(option_id=question.options[0].id)
```

API method:

```text
POST /api/game/{session_id}/answer
```

Answer via the exact option text:

```python
result = game.answer_by_text("Paris")
```

Warning: `answer_by_text` looks for a match between the passed text and the option text. It is generally more robust to use `option_id` directly.

## Minimal game loop

```python
def choose_answer(question):
    # Baseline: always picks the first option.
    # To be replaced with the model.
    return question.options[0].id

game = client.game.start(competition_id=1)

while game.in_progress:
    question = game.current_question
    if question is None:
        break

    print(f"Level {game.current_level}")
    print(question.text)
    for opt in question.options:
        print(f"{opt.id}: {opt.text}")

    option_id = choose_answer(question)
    result = game.answer(option_id)

    print("Correct:", result.correct)
    print("Game over:", result.game_over)
    print("Earned:", result.earned_amount)
```

## Integration with a model

The key function to implement is a strategy that takes a `Question` and returns an `option_id`.

```python
def answer_strategy(question):
    prompt = f"""
Question: {question.text}
Options:
{chr(10).join(f"{opt.id}. {opt.text}" for opt in question.options)}

Return only the id of the correct option.
"""

    predicted_id = run_local_model(prompt)
    return int(predicted_id)
```

In the real project it is best to validate the output:

```python
def safe_answer_strategy(question):
    predicted_id = answer_strategy(question)
    valid_ids = {opt.id for opt in question.options}

    if predicted_id not in valid_ids:
        return question.options[0].id

    return predicted_id
```

## Leaderboard

To read the leaderboard:

```python
lb = client.leaderboard.get(competition_id=1, limit=10)

for i, entry in enumerate(lb.entries, 1):
    print(i, entry.username, entry.score, entry.reached_level)
```

For the speech leaderboard:

```python
lb = client.leaderboard.get(competition_id=1, limit=10, mode="speech")
```

API method:

```text
GET /api/leaderboard/{competition_id}?limit=10&mode=text
GET /api/leaderboard/{competition_id}?limit=10&mode=speech
```

Main fields of an entry:

```text
entry.username
entry.score
entry.reached_level
entry.finished_at
entry.total_trials
```

## Errors handled by the client

The package defines these exceptions:

```python
from millionaire_client import (
    MillionaireError,
    AuthenticationError,
    GameError,
    TimeoutError,
    ValidationError,
    NotFoundError,
    ServerError,
    RateLimitError,
)
```

Recommended usage:

```python
try:
    result = game.answer(option_id)
except TimeoutError:
    print("Timeout on current question")
except MillionaireError as e:
    print("API error:", e)
```

## Recommended logging for the project

To be able to do a good final analysis, save at least:

```text
session_id
competition_id
question_id
level
question_text
options
predicted_option_id
predicted_option_text
correct
earned_amount
time_remaining_before_answer
latency_seconds
model_name
strategy_name
prompt_version
retrieved_context
error_message
```

For speech runs also add:

```text
mode
audio_question_path
audio_option_paths_json
audio_fetch_latency_seconds
time_remaining_after_audio
```

Example:

```python
import time

logs = []

start = time.time()
option_id = safe_answer_strategy(question)
latency = time.time() - start
time_left = game.time_remaining

result = game.answer(option_id)

logs.append({
    "session_id": game.session_id,
    "level": game.current_level,
    "question_id": question.id,
    "question_text": question.text,
    "options": [(opt.id, opt.text) for opt in question.options],
    "predicted_option_id": option_id,
    "correct": result.correct,
    "earned_amount": result.earned_amount,
    "latency_seconds": latency,
    "time_remaining_before_answer": time_left,
    "mode": game.mode,
    "strategy_name": "baseline_first_option",
})
```

## Practical notes

- Do not make many consecutive requests too quickly: the assignment explicitly asks to avoid excessive load on the server.
- Answering within 30 seconds is part of the task: always measure the model latency.
- In `speech` mode, measure separately the audio fetch time and the model latency; the timer starts after the last audio option.
- The client audio is WAV returned as complete bytes, not streaming; save it to disk before playing or analyzing it.
- Use `option_id` and not the text when possible.
- Keep passwords and tokens out of the shared notebook; on Colab use the secrets.
- The delivered notebook must clearly explain the model, prompt, retrieval, evaluation, and limitations.
