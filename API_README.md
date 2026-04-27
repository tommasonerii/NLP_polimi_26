# README API PoliMillionaire

Questo file spiega come usare il client Python fornito nel pacchetto `NLP_assignment_api_client.zip` per interagire con il gioco "Who wants to be a PoliMillionaire?".

## Dove si trova il client

Il client e nello zip:

```text
docs/Tutorials-20260427/NLP_assignment_api_client.zip
```

Dentro lo zip ci sono:

```text
NLP_assignment_api_client/
|-- PoliMillionaire.ipynb
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

Nel progetto finale, in Colab, la cartella `millionaire_client` deve stare nello stesso parent directory del notebook, oppure quel parent directory deve essere aggiunto a `sys.path`.

Esempio consigliato su Google Drive:

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

## URL del server

La consegna indica questo endpoint:

```python
API_URL = "http://131.175.15.22:51111/"
```

Nota: la consegna dice che il sito potrebbe non essere accessibile dalla rete Wi-Fi PoliMi per un blocco sulla porta.

## Registrazione e login

Prima bisogna registrarsi dal browser sul sito:

```text
http://131.175.15.22:51111/
```

Poi nel notebook si usa il login:

```python
client = MillionaireClient(API_URL)

try:
    user = client.login(username, password)
    print(f"Welcome, {user.username}! Role: {user.role}")
except AuthenticationError as e:
    print(f"Login failed: {e}")
```

Metodo usato internamente:

```text
POST /api/auth/login
```

Il client salva automaticamente il cookie di autenticazione nella sessione `requests`.

## Oggetto principale: MillionaireClient

`MillionaireClient` espone quattro moduli:

```python
client.auth          # autenticazione
client.competitions  # competizioni disponibili
client.game          # partite e risposte
client.leaderboard   # classifiche
```

Ha anche shortcut comodi:

```python
client.login(username, password)
client.logout()
client.user
client.is_authenticated
```

## Competizioni

Per vedere le competizioni disponibili:

```python
competitions = client.competitions.list_all()

for comp in competitions:
    print(comp.id, comp.name, comp.max_levels)
```

Metodo API:

```text
GET /api/competitions
```

Per ottenere la configurazione dettagliata:

```python
config = client.competitions.get_config(competition_id=1)
print(config.name)
print(config.max_levels)
print(config.money_pyramid)
```

Metodo API:

```text
GET /api/competitions/{competition_id}/config
```

## Avviare una partita

```python
comp_id = 1
game = client.game.start(competition_id=comp_id)

print(game.session_id)
print(game.current_level)
print(game.earned_amount)
```

Metodo API:

```text
POST /api/game/start
```

Il risultato e un oggetto `GameSession`.

## Leggere la domanda corrente

```python
question = game.current_question

print(question.id)
print(question.level)
print(question.text)

for opt in question.options:
    print(opt.id, opt.text)
```

Campi principali:

```text
question.id       # id domanda
question.text     # testo domanda
question.level    # livello
question.options  # lista di Option
```

Ogni `Option` ha:

```text
opt.id
opt.text
```

## Tempo disponibile

Ogni domanda ha un timeout, indicato nella consegna come massimo 30 secondi.

```python
time_left = game.time_remaining
print(time_left)
```

Se si risponde troppo tardi, il server puo restituire timeout anche se l'opzione scelta era corretta.

## Rispondere

Risposta tramite id dell'opzione:

```python
result = game.answer(option_id=question.options[0].id)
```

Metodo API:

```text
POST /api/game/{session_id}/answer
```

Risposta tramite testo esatto dell'opzione:

```python
result = game.answer_by_text("Paris")
```

Attenzione: `answer_by_text` cerca una corrispondenza tra il testo passato e il testo delle opzioni. In genere e piu robusto usare direttamente `option_id`.

## Ciclo partita minimo

```python
def choose_answer(question):
    # Baseline: sceglie sempre la prima opzione.
    # Da sostituire con il modello.
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

## Integrazione con un modello

La funzione chiave da implementare e una strategia che prende una `Question` e restituisce un `option_id`.

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

Nel progetto reale conviene validare l'output:

```python
def safe_answer_strategy(question):
    predicted_id = answer_strategy(question)
    valid_ids = {opt.id for opt in question.options}

    if predicted_id not in valid_ids:
        return question.options[0].id

    return predicted_id
```

## Leaderboard

Per leggere la classifica:

```python
lb = client.leaderboard.get(competition_id=1, limit=10)

for i, entry in enumerate(lb.entries, 1):
    print(i, entry.username, entry.score, entry.reached_level)
```

Metodo API:

```text
GET /api/leaderboard/{competition_id}
```

Campi principali di una entry:

```text
entry.username
entry.score
entry.reached_level
entry.finished_at
entry.total_trials
```

## Errori gestiti dal client

Il pacchetto definisce queste eccezioni:

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

Uso consigliato:

```python
try:
    result = game.answer(option_id)
except TimeoutError:
    print("Timeout on current question")
except MillionaireError as e:
    print("API error:", e)
```

## Logging consigliato per il progetto

Per poter fare una buona analisi finale, salvare almeno:

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

Esempio:

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
    "strategy_name": "baseline_first_option",
})
```

## Note pratiche

- Non fare molte richieste consecutive troppo velocemente: la consegna chiede esplicitamente di evitare carichi eccessivi sul server.
- Rispondere entro 30 secondi e parte del task: misurare sempre la latenza del modello.
- Usare `option_id` e non il testo quando possibile.
- Tenere password e token fuori dal notebook condiviso; in Colab usare i secrets.
- Il notebook consegnato deve spiegare chiaramente modello, prompt, retrieval, valutazione e limiti.
