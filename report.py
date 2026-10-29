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
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


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
        for b, v, l0 in zip(bars, vals, left):
            if v:
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
    drift_w = sum(1 for r in results if r["with"])
    drift_wo = sum(1 for r in results if r["without"])
    high = sum(1 for r in results if any(f["sev"] == "high" for f in r["with"]))
    traps = sum(1 for r in results for f in r["with"] if f["type"] == "language_trap")
    cells = [
        (f"{drift_w} of {n}", "questions where Meta's model said something different in "
                              "Portuguese (with the interpreter trap file)"),
        (f"{high}", "questions with at least one HIGH-severity difference "
                    "(could cause harm, a wrong dose, a missed deadline or a lost right)"),
        (f"{drift_wo} → {drift_w}", "questions flagged without → with the interpreter trap file"),
        (f"{traps}", "interpreter-trap flags (false friends, calques, register) "
                     "that only the trap file catches"),
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


def findings_table(results, st):
    rows = []
    for r in results:
        for f in r["with"]:
            rows.append((SEV_ORDER.get(f["sev"], 3), r["id"], r["domain"], f))
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

    table, n = findings_table(results, st)
    story += [Paragraph(f"What went wrong: all {n} flags (with the interpreter trap file), "
                        "most serious first", st["h2"]),
              Paragraph("Each row quotes the exact sentence from each answer. These flags come "
                        "from the automatic judge and still need the interpreter's review "
                        "(real or false alarm) before they count toward the accuracy score.",
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
