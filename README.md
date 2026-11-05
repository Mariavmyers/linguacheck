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

▶️ **Demo video (2:47):** https://youtu.be/X9qs2wmBoCU · **Live app:** https://huggingface.co/spaces/hugmarie/linguacheck

**Benchmarks ask *"is the AI right?"* LinguaCheck asks *"does the AI say the same thing in every language?"***

LinguaCheck is a **cross-lingual factual consistency evaluator** for Meta's AI models. It asks the same medical, legal or finance question in **English** and **Brazilian Portuguese**, compares the two answers, and flags **cross-lingual drift**: places where the model gives different facts, warnings, numbers, deadlines or rights depending on the language it is asked in. A professional interpreter then reviews the flags.

> Model output shown in LinguaCheck is **output under audit — not medical, legal or financial advice.** All test cases are fictional; no real patient or client data is used anywhere.

## Built during the hackathon
The code, the app, the test set and the review were created **Oct 26 – Nov 4, 2026** for the Meta Global AI Developer Hackathon (Indie Developer track, "Build for Your World"). The code was written with an AI coding assistant (Claude); the test design, interpreter traps and review decisions are by Maria Myers, a professional medical & legal interpreter (EN / PT-BR / ES, about 20 years).

## How it works
1. **Ask** — the same question goes to **Muse Spark Open** in English and in Brazilian Portuguese (temperature 0).
2. **Check, layer 1 (code)** — numbers, units and dates are compared exactly, aware of the Brazilian decimal comma (2,5 = 2.5), thousands dot and date order.
3. **Check, layer 2 (AI judge)** — the same Muse Spark Open model, in a second evaluation step and guided by an interpreter trap file (`data/traps.csv`), lists each meaningful difference, quotes the evidence from both answers and assigns a severity.
4. **Human review** — the interpreter confirms or rejects each flag, can add errors the judge missed, and writes a corrected version. Review decisions are saved separately from the model's answers, which are never edited.

No answer key is needed to *flag* drift: the model is compared with itself. Human review decides which flags are real and which language is wrong.

## Meta model used
- **Muse Spark Open** (Meta open weights) served by **Baseten** — `meta-models/muse-spark-open-1026`, OpenAI-compatible endpoint `https://inference.baseten.co/v1`. Both the answers and the judge use this model.

## Results (recorded run, Oct 29–30, 2026)
| | |
|---|---|
| Test questions | 75 (25 medical, 25 legal, 25 finance), fictional and deliberately difficult |
| Flags raised by the AI judge | 143 (45 high, 87 medium, 11 low) |
| Flags reviewed by the interpreter | 50 — all 45 high-severity flags plus 5 medium |
| Confirmed | **47 of 50 reviewed flags (94%)**; 3 rejected as false alarms |
| Questions with at least one confirmed problem | 28 of 75 |
| Wrong language (confirmed) | Portuguese 43 · English 4 |
| By domain (confirmed) | Finance 23 · Legal 15 · Medical 9 |
| Severity after review | 20 high · 24 medium · 3 low |
| Errors the judge missed, found by the interpreter | 3 (L14, L19, F18), not counted in the 47 |

**Limits — please read:**
- The 94% is the confirmation rate among the 50 **reviewed** flags, and that review prioritized high-severity flags. It is not the overall precision or accuracy of LinguaCheck. 93 flags have not been reviewed and are not counted.
- **Recall is not measured**: answers without flags were not systematically checked for missed errors.
- 75 deliberately difficult questions are a small test set. The results show that the problem is real; they are **not a general error rate** for the model.
- The review is a **linguistic review by a professional interpreter**, who is not a physician, attorney or accountant. Sources are listed in the app where they were checked.
- Some reviewer verdicts and corrected versions were drafted with AI assistance and approved by the interpreter.
- Temperature 0 reduces but does not guarantee identical outputs on a re-run.

## Using the app
- **★ Confirmed errors** — the reviewed findings, each with Muse's English and Portuguese wording, the reviewer verdict, a corrected version and the source. Muse's original answers are stored unchanged; the cards show excerpts selected for review. For omissions, the card shows what the answer said on that topic instead.
- **1 · Ask one question** — runs one test question live in both languages and judges it.
- **2 · Review findings, 3 · Results, 4 · Runs** — the owner's review and full-run tools (hidden in public mode).

### Public mode
Set the Space variable `PUBLIC_MODE=1` when the app is public. Visitors then see only **Confirmed errors** and **Ask one question** (limited to `PUBLIC_DAILY_LIMIT`, default 30 live questions per day, chosen from the test set). Saving reviews, full runs, re-judging and exports are refused on the server, not just hidden.

## Run it on your computer
1. Install Python 3.10 or newer.
2. In this folder: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and paste your Baseten key after `BASETEN_API_KEY=` (`.env` never goes to GitHub).
4. `python app.py`, then open the link it prints (usually http://127.0.0.1:7860).
5. Tests: `python -m pytest tests`

To audit the whole test set with and without the trap file: `python run_all.py` (or `python run_all.py M01 L03` for a few questions).

## Run it on Hugging Face Spaces
Create a Gradio Space from this repository and add `BASETEN_API_KEY` under **Settings → Variables and secrets → New secret**. Never put a key in the code. To keep runs and reviews permanently, add an `HF_TOKEN` secret with write access only to a **private** dataset (default `hugmarie/linguacheck-data`). The app runs on CPU; it includes a placeholder `@spaces.GPU` function only so it can start on Hugging Face's free ZeroGPU hardware.

## Next steps (not done yet)
- Review the remaining 93 flags and measure recall on unflagged answers.
- An "uncertain" label when the judge cannot tell which language is wrong.
- More languages, especially low-resource ones; Spanish is planned but not yet tested.
- Use the reviewed correction pairs to fine-tune a small model (not done; correction pairs only).

## License
MIT
