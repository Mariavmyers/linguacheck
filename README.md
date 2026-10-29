---
title: LinguaCheck
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# LinguaCheck

**Benchmarks ask *"is the AI right?"* LinguaCheck asks *"does the AI say the same thing in every language?"***

LinguaCheck is a **cross-lingual factual consistency evaluator** for Meta's AI models. It asks the same medical or legal question in **English** and **Brazilian Portuguese**, compares the answers, and flags **cross-lingual drift**: places where the model gives different facts, warnings, numbers, deadlines, or rights depending on the language it is asked in.

> Model output shown in LinguaCheck is **output under audit — not medical or legal advice.** LinguaCheck tests the AI's medical and legal *language*; it does not advise anyone.

## Built during the hackathon
Everything in this repository — the code, the app, and the expert-written test set — was created **Oct 26 – Nov 4, 2026** for the Meta Global AI Developer Hackathon (Indie Developer track, "Build for Your World").

## How it works
1. **Ask** — the same question goes to a Meta model in English and in Brazilian Portuguese.
2. **Normalize** — key facts are pulled out of each answer (doses, units, warnings, deadlines, rights).
3. **Compare** — a Meta model acting as judge lists every fact that differs, quotes the evidence, and assigns a risk level.
4. **Verify** — an expert answer key, written by a 20-year professional medical & legal interpreter, measures how accurate the flags are.

**No answer key is needed to flag drift**: the model is compared with itself. The expert key then measures how accurate those flags are.

## Meta models used
- **Muse Spark** via the Meta Model API
- **Muse Spark Open** (open weights) via Baseten — `meta-models/muse-spark-open-1026`

## What makes it different
- Built around **interpreter traps**: false friends, decimal comma vs. point, abbreviations, business vs. calendar days, brand names used as generic product names, US legal concepts with no Brazilian equivalent.
- **High-stakes medical & legal language**, in Brazilian Portuguese — an area existing research barely tests.
- Ends with **candidate correction pairs** (model answer, expert answer, why), not just a score.

## Test set
75 fictional, expert-written questions (25 medical, 25 legal and 25 finance). No real patient or client data is used anywhere.

## Run it on your computer
1. Install Python 3.10 or newer.
2. In this folder: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and paste your Baseten key after `BASETEN_API_KEY=` (`.env` never goes to GitHub).
4. `python app.py`, then open the link it prints (usually http://127.0.0.1:7860).

Every answer and judgment is saved to `results/answers.jsonl` (kept on your computer only).

To audit the whole test set with and without the trap file: `python run_all.py` (or `python run_all.py M01 L03` for a few questions).

## Run it on Hugging Face Spaces
Create a Gradio Space from this repository and add `BASETEN_API_KEY` under **Settings → Variables and secrets → New secret**. Never put the key in the code.

## Status
🚧 In progress during the hackathon.

- ✅ Ask Muse Spark Open the same question in EN and PT-BR (temperature 0) and show both final answers side by side with the interpreter's answer key
- ✅ Judge, layer 1 (code): exact comparison of numbers and dates, aware of the Brazilian decimal comma (2,5 = 2.5), thousands dot (1.200 = 1,200) and US vs. Brazilian date order
- ✅ Judge, layer 2 (Meta model): finds fact drift, missing facts and interpreter traps, quoting the exact EN and PT-BR sentences as evidence, with a severity level
- ✅ Interpreter trap file (`data/traps.csv`): 82 traps (25 medical, 25 legal, 25 finance, plus 7 confirmed rules), switchable on/off to measure what interpreter knowledge adds
- ✅ `python run_all.py`: runs the whole test set with and without the trap file and writes a review sheet for the expert to mark each flag "real" or "false alarm"
- ⏳ Next: run on the full set, measure accuracy against the expert's labels, add Muse Spark via the Meta Model API

## License
MIT
