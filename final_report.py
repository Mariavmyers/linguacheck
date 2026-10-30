"""
Final LinguaCheck results report (expert-confirmed findings only).

Usage:
  python final_report.py RUN.jsonl reviews.json expert_added.json translations.json OUT.pdf

Inputs are the saved run (model answers + judge flags), the interpreter's verdicts,
her extra findings the judge missed, and English translations of Portuguese text.
"""

import json
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from report import (BLUE, CRITICAL, GRID, INK2, SERIOUS, SEV_COLOR, SEV_LABEL, SEV_ORDER,
                    WARNING, esc, fig_png, style_axes, styles)

FEATURED = ["L07-31a066e75a", "M06-fb23f38106", "F03-933b4e776f", "M21-183f4f9665",
            "L14-9d4ffc5809"]
CAT = {"contradiction": "Contradiction", "changed_value": "Changed number / value",
       "consequential_omission": "Consequential omission",
       "terminology_register": "Terminology / register",
       "legal_overgeneralization": "Legal overgeneralization"}
LANG = {"PT": "Portuguese", "EN": "English", "both": "Both", "unclear": "Unclear"}
DOMAINS = ["Medical", "Legal", "Finance"]


def missing(q):
    return not q or q.strip().lower() in ("(missing)", "missing")


def load(run_path, rev_path, exp_path, tr_path):
    records = [json.loads(line) for line in open(run_path, encoding="utf-8") if line.strip()]
    reviews = json.load(open(rev_path, encoding="utf-8"))
    expert = json.load(open(exp_path, encoding="utf-8"))
    tr = json.load(open(tr_path, encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    confirmed, false_alarms, flags_total, high_flags = [], 0, 0, 0
    for r in records:
        for f in r["with_traps"]["findings"]:
            if f.get("support") == "supported" and f["category"] in (
                    "contradiction", "changed_value", "consequential_omission",
                    "terminology_register"):
                flags_total += 1
                high_flags += f["severity"] == "high"
            rev = reviews.get(f.get("id"))
            if not rev:
                continue
            if rev["decision"] == "Confirmed issue":
                confirmed.append((r, f, rev))
            elif rev["decision"].startswith("False"):
                false_alarms += 1
    exp = [(k, by_id.get(e["question_id"], {}), e) for k, e in expert.items()]
    return dict(records=records, confirmed=confirmed, false_alarms=false_alarms,
                flags_total=flags_total, high_flags=high_flags, expert=exp, tr=tr,
                questions_flagged=len({r["id"] for r in records
                                       for f in r["with_traps"]["findings"]
                                       if f.get("support") == "supported"}))


def sev(f, rev):
    return rev.get("severity") or f["severity"]


def chart_domain(confirmed):
    counts = {d: Counter() for d in DOMAINS}
    for r, f, rev in confirmed:
        counts[r["domain"]][sev(f, rev)] += 1
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    left = [0] * len(DOMAINS)
    for s in ("high", "medium", "low"):
        vals = [counts[d][s] for d in DOMAINS]
        bars = ax.barh(DOMAINS, vals, left=left, color=SEV_COLOR[s], label=SEV_LABEL[s],
                       height=0.55, edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, str(v),
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + 0.4, i, f"{total}", va="center", fontsize=10, color=INK2)
    ax.invert_yaxis()
    style_axes(ax)
    ax.set_xlabel("Confirmed problems")
    ax.set_xlim(0, max(left) + 3)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=8)
    ax.set_title("Confirmed problems by domain and severity", loc="left", fontsize=11)
    return fig_png(fig)


def chart_language(confirmed):
    c = Counter(rev.get("problem_language", "unclear") for _, _, rev in confirmed)
    labels = ["Portuguese answer", "English answer"]
    vals = [c["PT"], c["EN"]]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars = ax.bar(labels, vals, color=[BLUE, GRID], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=12,
                fontweight="bold", color="#0b0b0b")
    style_axes(ax)
    ax.set_ylim(0, max(vals) + 7)
    ax.set_title("Which answer had the problem", loc="left", fontsize=11)
    return fig_png(fig)


def quote_cell(text, translation, st, wrong):
    if missing(text):
        body = ("<i>Not mentioned in this answer.</i>" if wrong
                else "<i>No problem quoted on this point.</i>")
    else:
        body = esc(text)
        if translation:
            body += f"<br/><font color='{INK2}'><i>English translation: {esc(translation)}</i></font>"
    return Paragraph(body, st["cell"])


def case_block(r, f, rev, tr, st, expert=False):
    lang = rev.get("problem_language", "unclear")
    s = rev.get("severity") or f.get("severity")
    cat = rev.get("category") or f.get("category")
    head = (f"<b>{esc(r.get('id', ''))} · {esc(r.get('domain', ''))} · "
            f"<font color='{SEV_COLOR.get(s, INK2)}'>{SEV_LABEL.get(s, s)}</font> · "
            f"{esc(CAT.get(cat, cat))} · wrong answer: {LANG.get(lang, lang)}</b>"
            + (" · <i>missed by the AI judge, caught by the expert</i>" if expert else ""))
    en = f.get("en_quote") if not expert else f.get("evidence_en")
    pt = f.get("pt_quote") if not expert else f.get("evidence_pt")
    t = tr or {}
    grid = Table([[Paragraph("<b>Muse in English</b>", st["cellb"]),
                   Paragraph("<b>Muse in Portuguese</b>", st["cellb"])],
                  [quote_cell(en, "", st, lang in ("EN", "both")),
                   quote_cell(pt, t.get("pt_quote_en"), st, lang in ("PT", "both"))]],
                 colWidths=[4.85 * inch, 4.85 * inch])
    red = colors.HexColor(CRITICAL)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f2ee"))]
    if lang in ("EN", "both"):
        style.append(("BOX", (0, 1), (0, 1), 1.2, red))
    if lang in ("PT", "both"):
        style.append(("BOX", (1, 1), (1, 1), 1.2, red))
    grid.setStyle(TableStyle(style))
    notes = rev.get("notes", "").replace(" FEATURED DEMO CASE.", "")
    correct = rev.get("correct_version", "")
    parts = [Paragraph(head, st["body"]),
             Paragraph(f"<b>Question:</b> {esc(r.get('question_en', ''))}", st["cell"]),
             Spacer(1, 3), grid, Spacer(1, 3),
             Paragraph(f"<b>Expert verdict:</b> {esc(notes)}"
                       + (f"<br/><font color='{INK2}'><i>English: {esc(t['notes_en'])}</i></font>"
                          if t.get("notes_en") else ""), st["cell"])]
    if correct:
        parts.append(Paragraph(
            f"<b>Correct version:</b> {esc(correct)}"
            + (f"<br/><font color='{INK2}'><i>English: {esc(t['correct_version_en'])}</i></font>"
               if t.get("correct_version_en") else ""), st["cell"]))
    parts += [Paragraph(f"<font color='{INK2}'>Source: {esc(rev.get('source', ''))}</font>",
                        st["cell"]), Spacer(1, 10)]
    return KeepTogether(parts)


def build(data, out_path, model="Muse Spark Open (Meta, open weights) via Baseten"):
    st = styles()
    conf = data["confirmed"]
    n = len(data["records"])
    decided = len(conf) + data["false_alarms"]
    qs = len({r["id"] for r, _, _ in conf})
    sevc = Counter(sev(f, rev) for _, f, rev in conf)
    doc = SimpleDocTemplate(out_path, pagesize=landscape(letter),
                            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="LinguaCheck results", author="Maria Myers · LinguaCheck")
    heroes = [(f"{n}", "high-stakes questions (medical, legal, finance), each asked in English "
                       "and in Brazilian Portuguese"),
              (f"{len(conf)}", f"confirmed cross-language problems: {sevc['high']} high, "
                               f"{sevc['medium']} medium, {sevc['low']} low"),
              (f"{qs} of {n}", f"questions ({qs / n:.0%}) with at least one confirmed problem"),
              (f"{len(conf) / decided:.0%}", f"of the {decided} automated flags the expert "
                                             "decided were real (judge precision)")]
    hero = Table([[Paragraph(v, st["hero"]) for v, _ in heroes],
                  [Paragraph(esc(lab), st["herolab"]) for _, lab in heroes]],
                 colWidths=[2.45 * inch] * 4)
    hero.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.HexColor(BLUE)),
                              ("TOPPADDING", (0, 0), (-1, 0), 8),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story = [
        Paragraph("LinguaCheck: does Meta's AI say the same thing in English and Portuguese?",
                  st["title"]),
        Paragraph(f"Model under audit: {esc(model)} · temperature 0 · one attempt per question "
                  f"(pass@1) · expert review by Maria Myers, professional medical &amp; legal "
                  f"interpreter (EN/PT-BR) · {datetime.now().strftime('%B %d, %Y')}", st["muted"]),
        Spacer(1, 12), hero, Spacer(1, 14),
        Table([[Image(chart_domain(conf), width=5.6 * inch, height=2.6 * inch),
                Image(chart_language(conf), width=4.0 * inch, height=2.5 * inch)]]),
        Spacer(1, 10),
        Paragraph("<b>How it works.</b> LinguaCheck asks Meta's Muse the same question in English "
                  "and in Brazilian Portuguese, then a Muse judge, guided by an interpreter's trap "
                  "file, flags meaningful differences with exact quotes from both answers. A "
                  "professional interpreter then confirms or rejects each flag against official "
                  f"sources. {data['flags_total']} automated flags were raised; the expert reviewed "
                  f"all {data['high_flags']} high-severity flags plus 5 others. She also found "
                  f"<b>{len(data['expert'])} errors the AI judge missed</b>.", st["body"]),
        Spacer(1, 6),
        Paragraph("<b>Honest limits.</b> All test questions are fictional; model output is under "
                  "audit, not advice. Unreviewed flags are not counted. Answers without flags "
                  "were not checked for missed errors (recall is not measured). Rules that vary "
                  "by state are not presented as universal.", st["muted"]),
        PageBreak(),
        Paragraph("Featured cases", st["h2"]),
    ]
    by_id = {f["id"]: (r, f, rev) for r, f, rev in conf}
    for fid in FEATURED:
        if fid in by_id:
            r, f, rev = by_id[fid]
            story.append(case_block(r, f, rev, data["tr"].get(fid), st))
    story.append(Paragraph("Caught by the expert, missed by the AI judge", st["h2"]))
    for k, r, e in data["expert"]:
        story.append(case_block(r, e, e, data["tr"].get(k), st, expert=True))
    story += [PageBreak(), Paragraph(f"All {len(conf)} confirmed problems", st["h2"])]
    rows = [[Paragraph(f"<b>{h}</b>", st["cellb"]) for h in
             ("ID", "Domain", "Severity", "Type", "Wrong", "Expert verdict", "Source")]]
    for r, f, rev in sorted(conf, key=lambda c: (SEV_ORDER.get(sev(c[1], c[2]), 3),
                                                  c[0]["domain"], c[0]["id"])):
        s = sev(f, rev)
        cat = rev.get("category") or f["category"]
        rows.append([Paragraph(esc(r["id"]), st["cell"]), Paragraph(r["domain"], st["cell"]),
                     Paragraph(f"<font color='{SEV_COLOR[s]}'><b>{SEV_LABEL[s]}</b></font>",
                               st["cell"]),
                     Paragraph(esc(CAT.get(cat, cat)), st["cell"]),
                     Paragraph(LANG.get(rev.get("problem_language"), "?"), st["cell"]),
                     Paragraph(esc(rev.get("notes", "").replace(" FEATURED DEMO CASE.", "")), st["cell"]),
                     Paragraph(esc(rev.get("source", "")), st["cell"])])
    t = Table(rows, colWidths=[0.5, 0.65, 0.75, 1.15, 0.95, 4.15, 1.75], repeatRows=1)
    t._argW = [w * inch for w in t._argW]
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(GRID)),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f2ee"))]))
    story.append(t)
    doc.build(story)
    return out_path


if __name__ == "__main__":
    a = sys.argv[1:]
    build(load(a[0], a[1], a[2], a[3]), a[4])
    print("Wrote", a[4])
