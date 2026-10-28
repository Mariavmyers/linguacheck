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

## Status
🚧 In progress during the hackathon.

## License
MIT
