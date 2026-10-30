"""
Make the LinguaCheck results PDF from a full-run summary.

Usage:  python report.py results.json LinguaCheck_Results.pdf

results.json is the "Results (JSON)" box of the app's Full audit section
(or any file with the same shape: {"results": [{id, domain, risk, with, without}, ...]}).
"""

import json
import sys
from collections import Counter
from datetime import datetime
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import landscape, letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

# Validated reference palette (dataviz skill): slot 1 blue, slot 2 orange; status reds.
BLUE, ORANGE = "#2a78d6", "#eb6834"
CRITICAL, SERIOUS, WARNING = "#d03b3b", "#ec835a", "#fab219"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DOMAINS = ["Medical", "Legal", "Finance"]
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
SEV_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}
SEV_COLOR = {"high": CRITICAL, "medium": SERIOUS, "low": WARNING}
TYPE_LABEL = {"fact_drift": "Fact drift", "missing": "Missing in one language",
              "language_trap": "Interpreter trap"}

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DejaVu", FONT_DIR + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONT_DIR + "DejaVuSans-Bold.ttf"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="DejaVu", textColor=colors.HexColor(INK), alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="DejaVu-Bold",
                                fontSize=22, leading=26, alignment=TA_LEFT,
                                textColor=colors.HexColor(INK)),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=13, leading=17,
                             spaceBefore=6, spaceAfter=6, textColor=colors.HexColor(INK)),
        "body": ParagraphStyle("b", fontSize=9.5, leading=13, **base),
        "muted": ParagraphStyle("m", fontName="DejaVu", fontSize=8.5, leading=11,
                                textColor=colors.HexColor(INK2)),
        "cell": ParagraphStyle("c", fontName="DejaVu", fontSize=7.5, leading=9.5,
                               textColor=colors.HexColor(INK)),
        "cellb": ParagraphStyle("cb", fontName="DejaVu-Bold", fontSize=7.5, leading=9.5,
                                textColor=colors.HexColor(INK)),
        "hero": ParagraphStyle("h", fontName="DejaVu-Bold", fontSize=26, leading=30,
                               textColor=colors.HexColor(INK)),
        "herolab": ParagraphStyle("hl", fontName="DejaVu", fontSize=8.5, leading=11,
                                  textColor=colors.HexColor(INK2)),
    }


def esc(s):
    s = str(s or "").replace("**", "").replace("`", "")
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fig_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return buf


def style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)


def chart_by_domain(results):
    """Questions flagged per domain, with vs without the trap file."""
    w = {d: 0 for d in DOMAINS}
    wo = {d: 0 for d in DOMAINS}
    tot = Counter(r["domain"] for r in results)
    for r in results:
        if r["with"]:
            w[r["domain"]] += 1
        if r["without"]:
            wo[r["domain"]] += 1
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(DOMAINS))
    bw = 0.36
    b1 = ax.bar([x - bw / 2 - 0.01 for x in xs], [wo[d] for d in DOMAINS], bw,
                color=ORANGE, label="Without trap file")
    b2 = ax.bar([x + bw / 2 + 0.01 for x in xs], [w[d] for d in DOMAINS], bw,
                color=BLUE, label="With interpreter trap file")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2),
                        textcoords="offset points")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{d}\n(of {tot.get(d, 0)})" for d in DOMAINS])
    ax.set_ylabel("Questions with at least one flag")
    ymax = max([1] + list(w.values()) + list(wo.values()))
    ax.set_ylim(0, ymax * 1.25 + 0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    return fig_png(fig)


def chart_by_type(results):
    """All flags (with the trap file) by type, split by severity."""
    cnt = Counter((f["type"], f["sev"]) for r in results for f in r["with"])
    types = [t for t in TYPE_LABEL if any(cnt[(t, s)] for s in SEV_ORDER)] or list(TYPE_LABEL)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor(SURFACE)
    left = [0] * len(types)
    for sev in ("high", "medium", "low"):
        vals = [cnt[(t, sev)] for t in types]
        bars = ax.barh([TYPE_LABEL[t] for t in types], vals, left=left, height=0.55,
                       color=SEV_COLOR[sev], label=SEV_LABEL[sev],
                       edgecolor=SURFACE, linewidth=2)
        biggest = max(1, max(sum(cnt[(t, x)] for x in SEV_ORDER) for t in types))
        for b, v, l0 in zip(bars, vals, left):
            if v and v >= biggest * 0.06:
                ax.text(l0 + v / 2, b.get_y() + b.get_height() / 2, str(v), ha="center",
                        va="center", fontsize=8.5, color=INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlabel("Number of flags")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, max(left + [1]) * 1.15)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.45, -0.22), title="Severity", title_fontsize=8.5)
    return fig_png(fig)


def hero_table(results, st):
    n = len(results)
    differ = sum(1 for r in results if r["with"])
    contra = sum(1 for r in results if any(f["type"] == "fact_drift" for f in r["with"]))
    danger = sum(1 for r in results
                 if any(f["type"] == "fact_drift" and f["sev"] == "high" for f in r["with"]))
    drift_wo = sum(1 for r in results if r["without"])
    cells = [
        (f"{differ} of {n}", "questions where Meta's model's Portuguese answer differed from "
                             "its English answer (any flag)"),
        (f"{contra} of {n}", "questions where the two answers CONTRADICT each other "
                             "(different facts, numbers, rights or steps)"),
        (f"{danger}", "high-severity contradictions: could cause harm, delay care, "
                      "or cost someone a right"),
        (f"{drift_wo} → {differ}", "questions flagged without → with the interpreter trap file"),
    ]
    row = [[Paragraph(esc(v), st["hero"]), ] for v, _ in cells]
    lab = [[Paragraph(esc(l), st["herolab"])] for _, l in cells]
    data = [[r[0] for r in row], [l[0] for l in lab]]
    t = Table(data, colWidths=[2.45 * inch] * 4)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.HexColor(BLUE)),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t


TYPE_ORDER = {"fact_drift": 0, "language_trap": 1, "missing": 2}


def findings_table(results, st, only=None):
    rows = []
    for r in results:
        for f in r["with"]:
            if only and not only(f):
                continue
            rows.append(((TYPE_ORDER.get(f["type"], 3), SEV_ORDER.get(f["sev"], 3)),
                         r["id"], r["domain"], f))
    rows.sort(key=lambda x: (x[0], x[1]))
    head = ["ID", "Severity", "Type", "English answer says", "Portuguese answer says",
            "Why it matters"]
    data = [[Paragraph(h, st["cellb"]) for h in head]]
    for _, qid, dom, f in rows:
        sev = f["sev"]
        typ = TYPE_LABEL.get(f["type"], f["type"])
        if f.get("trap"):
            typ += f" ({f['trap']})"
        data.append([
            Paragraph(f"<b>{esc(qid)}</b><br/>{esc(dom)}", st["cell"]),
            Paragraph(f"<font color='{SEV_COLOR.get(sev, INK)}'>■</font> "
                      f"{esc(SEV_LABEL.get(sev, sev))}", st["cell"]),
            Paragraph(esc(typ), st["cell"]),
            Paragraph(esc(f["en"]), st["cell"]),
            Paragraph(esc(f["pt"]), st["cell"]),
            Paragraph(esc(f["why"]), st["cell"]),
        ])
    t = Table(data, colWidths=[0.7 * inch, 0.75 * inch, 1.15 * inch, 2.45 * inch,
                               2.45 * inch, 2.3 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t, len(rows)


def build(summary, out_path, model="Muse Spark Open (Meta, open weights) via Baseten"):
    results = sorted(summary["results"], key=lambda r: r["id"])
    st = styles()
    doc = SimpleDocTemplate(out_path, pagesize=landscape(letter),
                            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="LinguaCheck results", author="LinguaCheck")
    story = [
        Paragraph("LinguaCheck: does Meta's AI say the same thing in English and Portuguese?",
                  st["title"]),
        Paragraph(f"Full audit of {len(results)} interpreter-verified medical, legal and finance "
                  f"questions · Model: {esc(model)} · temperature 0 · "
                  f"{datetime.now().strftime('%B %d, %Y')}", st["muted"]),
        Paragraph("Model output under audit — not advice. All test cases are fictional.",
                  st["muted"]),
        Spacer(1, 14),
        hero_table(results, st),
        Spacer(1, 14),
    ]
    charts = Table([[Image(chart_by_domain(results), width=4.6 * inch, height=2.9 * inch),
                     Image(chart_by_type(results), width=4.6 * inch, height=2.9 * inch)],
                    [Paragraph("Questions flagged, by domain", st["muted"]),
                     Paragraph("What kind of difference (with the trap file), by severity",
                               st["muted"])]],
                   colWidths=[4.95 * inch, 4.95 * inch])
    story += [charts, PageBreak()]

    spot, ns = findings_table(results, st,
                              only=lambda f: f["type"] == "fact_drift" and f["sev"] == "high")
    story += [Paragraph(f"Spotlight: the {ns} most dangerous contradictions", st["h2"]),
              Paragraph("Same question, same model, different facts depending on the language. "
                        "A difference shows the answers cannot both be right; the interpreter "
                        "decides which one is wrong (sometimes it is the English).", st["muted"]),
              Spacer(1, 6), spot, PageBreak()]
    table, n = findings_table(results, st)
    story += [Paragraph(f"What went wrong: all {n} flags (with the interpreter trap file)",
                        st["h2"]),
              Paragraph("Order: contradictions first, then interpreter traps, then facts missing "
                        "in one language; most serious first within each. Each row quotes the exact "
                        "sentences. These flags come from the automatic judge and still need the "
                        "interpreter's review (real or false alarm) before they count toward the "
                        "accuracy score. 'Missing in one language' flags are the most likely to "
                        "include false alarms (extra detail rather than a real difference).",
                        st["muted"]),
              Spacer(1, 6), table, Spacer(1, 14)]

    clean = [r["id"] for r in results if not r["with"] and not r.get("judge_error")]
    unjudged = [r["id"] for r in results if r.get("judge_error")]
    errors = summary.get("errors") or []
    story += [Paragraph("Questions with no difference found", st["h2"]),
              Paragraph(esc(", ".join(clean)) or "None", st["body"])]
    if unjudged:
        story += [Spacer(1, 8), Paragraph("Not judged (the judge's reply could not be read)",
                                          st["h2"]), Paragraph(esc(", ".join(unjudged)), st["body"])]
    if errors:
        story += [Spacer(1, 8), Paragraph("Questions that could not be run", st["h2"]),
                  Paragraph(esc("; ".join(errors)), st["body"])]
    story += [Spacer(1, 10), Paragraph("How this was measured", st["h2"]), Paragraph(
        "Each question was asked once in English and once in Brazilian Portuguese, the same "
        "meaning in both, written by a professional medical and legal interpreter. Only the "
        "model's final answers were compared. Layer 1 (code) compared every number and date, "
        "aware of the Brazilian decimal comma and day/month order. Layer 2, the same Meta model "
        "acting as judge, listed every fact that differs and quoted the evidence. The judge "
        "ran twice: without and with the interpreter trap file (false friends, calques, "
        "register and brand-name traps verified by the interpreter).", st["body"])]
    doc.build(story)
    return out_path


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        build(json.load(f), sys.argv[2] if len(sys.argv) > 2 else "LinguaCheck_Results.pdf")


# ====================================================================== v2 report (honest metrics)

STATUS_COLOR = {"Confirmed issue": CRITICAL, "Needs specialist review": WARNING,
                "False alarm / acceptable variation": "#9a9893", "Not reviewed": "#d6d4ce"}
STATUS_SHORT = {"Confirmed issue": "Confirmed", "Needs specialist review": "Needs specialist",
                "False alarm / acceptable variation": "False alarm", "Not reviewed": "Not reviewed"}


def _v2_rows(records, reviews):
    import audit
    import judge
    by_ev = audit.review_index(records, reviews)
    rows = []
    for r in records:
        for f in r["with_traps"]["findings"]:
            if not judge.is_issue(f):
                continue
            rev = audit.review_for(r, f, reviews, by_ev) or {}
            cat, sev = audit.effective(f, rev)
            rows.append((r, f, rev, cat, sev, rev.get("decision", "Not reviewed")))
    return rows


def chart_review_status(rows):
    import judge
    cats = [c for c in judge.ISSUE_CATEGORIES]
    statuses = list(STATUS_COLOR)
    cnt = Counter((c, d) for _, _, _, c, _, d in rows)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor(SURFACE)
    left = [0] * len(cats)
    labels = [judge.CATEGORIES[c].replace("Changed number, date, unit, currency or jurisdiction",
                                          "Changed number/date/unit") for c in cats]
    biggest = max([1] + [sum(cnt[(c, s)] for s in statuses) for c in cats])
    for s in statuses:
        vals = [cnt[(c, s)] for c in cats]
        bars = ax.barh(labels, vals, left=left, height=0.55, color=STATUS_COLOR[s],
                       label=STATUS_SHORT[s], edgecolor=SURFACE, linewidth=2)
        for b, v, l0 in zip(bars, vals, left):
            if v and v >= biggest * 0.06:
                ax.text(l0 + v / 2, b.get_y() + b.get_height() / 2, str(v), ha="center",
                        va="center", fontsize=8.5, color=INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlabel("Automated flags")
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, biggest * 1.15)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4,
              bbox_to_anchor=(0.4, -0.22), title="Interpreter review", title_fontsize=8.5)
    return fig_png(fig)


def chart_domain_v2(records):
    import judge
    tot = Counter(r["domain"] for r in records)
    w = Counter(r["domain"] for r in records
                if any(judge.is_issue(f) for f in r["with_traps"]["findings"]))
    wo = Counter(r["domain"] for r in records
                 if any(judge.is_issue(f) for f in r["without_traps"]["findings"]))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(DOMAINS))
    bw = 0.36
    b1 = ax.bar([x - bw / 2 - 0.01 for x in xs], [wo[d] for d in DOMAINS], bw, color=ORANGE,
                label="Without trap file")
    b2 = ax.bar([x + bw / 2 + 0.01 for x in xs], [w[d] for d in DOMAINS], bw, color=BLUE,
                label="With interpreter trap file")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2),
                        textcoords="offset points")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{d}\n(of {tot.get(d, 0)})" for d in DOMAINS])
    ax.set_ylabel("Questions with ≥1 automated flag")
    ax.set_ylim(0, max([1] + list(w.values()) + list(wo.values())) * 1.3 + 0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    return fig_png(fig)


def build_v2(records, meta, reviews, out_path):
    import audit
    import judge
    st = styles()
    m = audit.metrics(records, reviews)
    w = m["modes"]["with"]
    rows = _v2_rows(records, reviews)
    doc = SimpleDocTemplate(out_path, pagesize=landscape(letter),
                            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="LinguaCheck report", author="LinguaCheck")
    n = m["questions_tested"]
    prec = (f"{w['precision']:.0%}" if w["precision"] is not None else "n/a")
    heroes = [
        (f"{n}", "questions tested, each asked once in English and once in Brazilian Portuguese"),
        (f"{w['questions_with_flags']} of {n}",
         "questions with at least one AUTOMATED flag (a possible difference, not a proven error)"),
        (f"{w['reviewed']} of {w['total_flags']}", "automated flags reviewed by the interpreter so far"),
        (f"{w['confirmed']}", f"confirmed issues in {w['questions_with_confirmed']} questions · "
                              f"{w['false_alarms']} false alarms · judge precision on decided flags: {prec}"),
    ]
    hero = Table([[Paragraph(esc(v), st["hero"]) for v, _ in heroes],
                  [Paragraph(esc(l), st["herolab"]) for _, l in heroes]],
                 colWidths=[2.45 * inch] * 4)
    hero.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.HexColor(BLUE)),
                              ("TOPPADDING", (0, 0), (-1, 0), 8),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story = [
        Paragraph("LinguaCheck: does Meta's AI say the same thing in English and Portuguese?",
                  st["title"]),
        Paragraph(f"Run {esc(meta.get('run_id'))} · answers from {esc(meta.get('answers_from_run'))} · "
                  f"{esc(meta.get('model'))} · temperature {esc(meta.get('temperature'))} · judge "
                  f"{esc(meta.get('judge_version'))} · created {esc(meta.get('created_utc'))} · "
                  f"report {datetime.now().strftime('%B %d, %Y')}", st["muted"]),
        Paragraph("Model output under audit — not advice. All test cases are fictional. Automated "
                  "flags are possible differences; only interpreter-confirmed findings are errors. "
                  "No flag does not mean both answers are correct. Recall is not measured.",
                  st["muted"]),
        Spacer(1, 12), hero, Spacer(1, 12),
        Table([[Image(chart_domain_v2(records), width=4.6 * inch, height=2.9 * inch),
                Image(chart_review_status(rows), width=4.6 * inch, height=2.9 * inch)],
               [Paragraph("Questions with automated flags, by domain (same saved answers judged "
                          "with and without the trap file)", st["muted"]),
                Paragraph("Automated flags by category and interpreter decision", st["muted"])]],
              colWidths=[4.95 * inch, 4.95 * inch]),
        PageBreak(),
    ]

    confirmed = [x for x in rows if x[5] == "Confirmed issue"]
    confirmed.sort(key=lambda x: (SEV_ORDER.get(x[4], 3), x[0]["id"]))
    story.append(Paragraph(f"Confirmed by the interpreter ({len(confirmed)})", st["h2"]))
    if not confirmed:
        story.append(Paragraph("No flags have been confirmed yet. Next step: review the "
                               "highest-severity flags in the app (tab 2).", st["body"]))
    else:
        data = [[Paragraph(h, st["cellb"]) for h in
                 ("ID", "Category / severity", "English answer says", "Portuguese answer says",
                  "Interpreter's reasoning and correct version")]]
        for r, f, rev, cat, sev, _ in confirmed:
            data.append([
                Paragraph(f"<b>{esc(r['id'])}</b><br/>{esc(r['domain'])}", st["cell"]),
                Paragraph(f"{esc(judge.CATEGORIES.get(cat, cat))}<br/>"
                          f"<font color='{SEV_COLOR.get(sev, INK)}'>■</font> {esc(SEV_LABEL.get(sev, sev))}"
                          f"<br/>Problem in: {esc(rev.get('problem_language') or f.get('problem_language'))}",
                          st["cell"]),
                Paragraph(esc(f["en_quote"]), st["cell"]),
                Paragraph(esc(f["pt_quote"]), st["cell"]),
                Paragraph(esc(rev.get("notes") or f["explanation"])
                          + (f"<br/><b>Correct:</b> {esc(rev['correct_version'])}"
                             if rev.get("correct_version") else "")
                          + (f"<br/><i>Source:</i> {esc(rev['source'])}" if rev.get("source") else ""),
                          st["cell"]),
            ])
        t = Table(data, colWidths=[0.75 * inch, 1.45 * inch, 2.45 * inch, 2.45 * inch, 2.7 * inch],
                  repeatRows=1)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(INK2)),
                               ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(GRID)),
                               ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(t)

    g = m["guide_effect"]
    wo = m["modes"]["without"]
    story += [Spacer(1, 12), Paragraph("Does the interpreter trap file help?", st["h2"]),
              Paragraph(
                  f"Same saved answers, judged twice. Without the trap file: {wo['total_flags']} flags "
                  f"in {wo['questions_with_flags']} questions. With it: {w['total_flags']} flags in "
                  f"{w['questions_with_flags']} questions. Flags found only with the trap file: "
                  f"{g['added_by_guide']['count']} (confirmed {g['added_by_guide']['confirmed']}, "
                  f"false alarms {g['added_by_guide']['false_alarms']}, not reviewed "
                  f"{g['added_by_guide']['not_reviewed']}). Flags found only without it: "
                  f"{g['removed_with_guide']['count']}. Same evidence, different category: "
                  f"{g['reclassified']}. More flags alone is not an improvement; what counts is how "
                  "many of the extra flags the interpreter confirms.", st["body"]),
              PageBreak()]

    rows.sort(key=lambda x: (list(STATUS_COLOR).index(x[5]), SEV_ORDER.get(x[4], 3), x[0]["id"]))
    data = [[Paragraph(h, st["cellb"]) for h in
             ("ID", "Status", "Category / severity", "English answer says",
              "Portuguese answer says", "Judge's explanation")]]
    for r, f, rev, cat, sev, d in rows:
        data.append([
            Paragraph(f"<b>{esc(r['id'])}</b><br/>{esc(r['domain'])}", st["cell"]),
            Paragraph(f"<font color='{STATUS_COLOR[d]}'>■</font> {esc(STATUS_SHORT[d])}", st["cell"]),
            Paragraph(f"{esc(judge.CATEGORIES.get(cat, cat))}<br/>{esc(SEV_LABEL.get(sev, sev))}",
                      st["cell"]),
            Paragraph(esc(f["en_quote"]), st["cell"]),
            Paragraph(esc(f["pt_quote"]), st["cell"]),
            Paragraph(esc(f["explanation"]), st["cell"]),
        ])
    t = Table(data, colWidths=[0.7 * inch, 0.95 * inch, 1.4 * inch, 2.3 * inch, 2.3 * inch, 2.15 * inch],
              repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(INK2)),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(GRID)),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [Paragraph(f"All {len(rows)} automated flags (with the trap file)", st["h2"]),
              Paragraph(f"Not counted as flags: {w['harmless_variation']} harmless variations, "
                        f"{w['insufficient_evidence']} with insufficient evidence, "
                        f"{w['unsupported_quotes']} whose quotes were not found word for word in the "
                        "saved answers.", st["muted"]),
              Spacer(1, 6), t]
    doc.build(story)
    return out_path
