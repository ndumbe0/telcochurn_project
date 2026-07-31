"""PDF report generator using ReportLab — single-customer churn report."""
import io
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette ─────────────────────────────────────────────────────────────
RED    = colors.HexColor("#e74c3c")
GREEN  = colors.HexColor("#2ecc71")
ORANGE = colors.HexColor("#f39c12")
BLUE   = colors.HexColor("#2980b9")
DARK   = colors.HexColor("#2c3e50")
LIGHT  = colors.HexColor("#ecf0f1")
MID    = colors.HexColor("#bdc3c7")
WHITE  = colors.white


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"],
            fontSize=22, textColor=DARK, spaceAfter=4, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=13, textColor=DARK, spaceBefore=14, spaceAfter=6,
            borderPad=2,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9, textColor=DARK, leading=14,
        ),
        "label": ParagraphStyle(
            "Label", parent=base["Normal"],
            fontSize=8, textColor=colors.grey,
        ),
        "bold": ParagraphStyle(
            "Bold", parent=base["Normal"],
            fontSize=9, textColor=DARK, fontName="Helvetica-Bold",
        ),
        "risk_high": ParagraphStyle(
            "RiskHigh", parent=base["Normal"],
            fontSize=14, textColor=RED, fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "risk_med": ParagraphStyle(
            "RiskMed", parent=base["Normal"],
            fontSize=14, textColor=ORANGE, fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "risk_low": ParagraphStyle(
            "RiskLow", parent=base["Normal"],
            fontSize=14, textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
    }
    return custom


def _kv_table(rows, col_widths=None):
    """Create a two-column key-value table."""
    w = col_widths or [6 * cm, 10 * cm]
    t = Table(rows, colWidths=w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("TEXTCOLOR",  (0, 0), (0, -1), DARK),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor("#f9f9f9")]),
        ("GRID",       (0, 0), (-1, -1), 0.5, MID),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return t


def _section(title, style_dict):
    return [
        Spacer(1, 0.3 * cm),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=4),
        Paragraph(title, style_dict["h2"]),
    ]


def generate_customer_pdf(
    customer_data: dict,
    prediction_result: dict,
    shap_values,
    feature_names,
    clv_info: dict,
    recommendations: list,
    model_name: str = "XGBoost",
) -> bytes:
    """
    Generate a single-customer churn analysis PDF.

    Returns bytes ready for st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    S = _styles()
    story = []

    prob = prediction_result.get("churn_probability", 0)
    pred = prediction_result.get("prediction", 0)
    label = prediction_result.get("churn_label", "No")
    risk_tag = "🔴 HIGH RISK" if prob > 0.6 else ("🟡 MEDIUM RISK" if prob > 0.3 else "🟢 LOW RISK")
    risk_style = "risk_high" if prob > 0.6 else ("risk_med" if prob > 0.3 else "risk_low")
    risk_color = RED if prob > 0.6 else (ORANGE if prob > 0.3 else GREEN)

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("📊 Telco Customer Churn Analysis Report", S["title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}  |  Model: {model_name}",
        S["subtitle"],
    ))

    # Risk banner
    banner_data = [[Paragraph(risk_tag, S[risk_style])]]
    banner = Table(banner_data, colWidths=[16 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(
            "#fde8e8" if prob > 0.6 else ("#fef3cd" if prob > 0.3 else "#d4edda")
        )),
        ("BOX", (0, 0), (-1, -1), 1.5, risk_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(banner)

    # ── Prediction summary ─────────────────────────────────────────────────────
    story += _section("1. Prediction Summary", S)
    summary_rows = [
        ["Churn Probability", f"{prob:.1%}"],
        ["Prediction", f"{'⚠ Will Churn' if pred == 1 else '✓ Will Stay'}"],
        ["Risk Level", risk_tag.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")],
        ["Active Model", model_name],
    ]
    story.append(_kv_table(summary_rows))

    # ── Customer profile ───────────────────────────────────────────────────────
    story += _section("2. Customer Profile", S)
    profile_keys = [
        ("Contract", "Contract"),
        ("tenure", "Tenure (months)"),
        ("MonthlyCharges", "Monthly Charges ($)"),
        ("TotalCharges", "Total Charges ($)"),
        ("InternetService", "Internet Service"),
        ("TechSupport", "Tech Support"),
        ("OnlineSecurity", "Online Security"),
        ("PaymentMethod", "Payment Method"),
        ("gender", "Gender"),
        ("SeniorCitizen", "Senior Citizen"),
        ("Partner", "Partner"),
        ("Dependents", "Dependents"),
    ]
    profile_rows = []
    for key, label_str in profile_keys:
        val = customer_data.get(key, "—")
        if key == "SeniorCitizen":
            val = "Yes" if val == 1 else "No"
        elif key in ("MonthlyCharges", "TotalCharges"):
            try:
                val = f"${float(val):,.2f}"
            except Exception:
                pass
        profile_rows.append([label_str, str(val)])

    # Split into two side-by-side columns
    half = len(profile_rows) // 2
    left = profile_rows[:half]
    right = profile_rows[half:]
    combined = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ["", ""]
        r = right[i] if i < len(right) else ["", ""]
        combined.append(l + r)

    t = Table(combined, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, MID),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor("#f9f9f9")]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # ── CLV ───────────────────────────────────────────────────────────────────
    if clv_info:
        story += _section("3. Customer Lifetime Value", S)
        clv_rows = [
            ["CLV Estimate", f"${clv_info.get('CLV Estimate ($)', 0):,.2f}"],
            ["Revenue at Risk", f"${clv_info.get('Revenue at Risk ($)', 0):,.2f}"],
            ["Est. Remaining Months", str(clv_info.get('Est. Remaining Months', '—'))],
        ]
        story.append(_kv_table(clv_rows))

    # ── SHAP top features ──────────────────────────────────────────────────────
    if shap_values is not None and feature_names is not None:
        story += _section("4. Key Churn Drivers (SHAP)", S)
        story.append(Paragraph(
            "The table below shows the top features influencing this prediction. "
            "Positive SHAP values increase churn risk; negative values reduce it.",
            S["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
        shap_header = [["Feature", "SHAP Value", "Direction"]]
        shap_rows = []
        for feat, val in pairs[:10]:
            direction = "↑ Increases churn risk" if val > 0 else "↓ Reduces churn risk"
            shap_rows.append([feat, f"{val:+.4f}", direction])

        shap_table_data = shap_header + shap_rows
        shap_t = Table(shap_table_data, colWidths=[6 * cm, 3 * cm, 7 * cm])
        shap_t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f0f4ff")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, MID),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(shap_t)

    # ── Intervention recommendations ───────────────────────────────────────────
    if recommendations:
        story += _section("5. Recommended Retention Actions", S)
        rec_header = [["Action", "New Churn Prob.", "Δ Reduction", "Effort"]]
        rec_rows = []
        for rec in recommendations[:6]:
            rec_rows.append([
                rec["title"].replace("📋 ", "").replace("🛠️ ", "").replace("🔒 ", "")
                             .replace("☁️ ", "").replace("💳 ", "").replace("💰 ", "")
                             .replace("🛡️ ", ""),
                f"{rec['new_prob']:.1%}",
                f"−{rec['delta_pct']:.1f} pp",
                rec.get("effort", "—"),
            ])

        rec_data = rec_header + rec_rows
        rec_t = Table(rec_data, colWidths=[7 * cm, 3 * cm, 3 * cm, 3 * cm])
        rec_t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  DARK),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f9f9f9")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, MID),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(rec_t)

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Paragraph(
        "This report was auto-generated by the Telco Churn Prediction System. "
        "Predictions are based on historical patterns and should be used as one input among many.",
        S["label"],
    ))

    doc.build(story)
    return buf.getvalue()
