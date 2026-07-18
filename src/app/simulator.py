"""What-if simulator: change individual features and see probability impact live."""
import plotly.graph_objects as go
from typing import List, Dict

WHAT_IF_FEATURES = {
    "Contract": {
        "type": "select",
        "options": ["Month-to-month", "One year", "Two year"],
        "label": "Contract Type",
    },
    "tenure": {
        "type": "slider",
        "min": 0, "max": 72, "step": 1,
        "label": "Tenure (months)",
    },
    "MonthlyCharges": {
        "type": "slider",
        "min": 18.0, "max": 120.0, "step": 1.0,
        "label": "Monthly Charges ($)",
    },
    "InternetService": {
        "type": "select",
        "options": ["DSL", "Fiber optic", "No"],
        "label": "Internet Service",
    },
    "TechSupport": {
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "label": "Tech Support",
    },
    "OnlineSecurity": {
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "label": "Online Security",
    },
    "PaymentMethod": {
        "type": "select",
        "options": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "label": "Payment Method",
    },
}


def build_what_if_chart(baseline_prob: float, scenarios: List[Dict]) -> go.Figure:
    """Bar chart comparing baseline vs. what-if scenario probabilities."""
    labels = ["📍 Baseline"] + [s["label"] for s in scenarios]
    probs = [baseline_prob * 100] + [s["probability"] * 100 for s in scenarios]
    deltas = [0.0] + [s["probability"] * 100 - baseline_prob * 100 for s in scenarios]

    colors = []
    for p in probs:
        if p > 60:
            colors.append("#e74c3c")
        elif p > 30:
            colors.append("#f39c12")
        else:
            colors.append("#2ecc71")

    hover = [
        f"Probability: {p:.1f}%<br>Δ vs baseline: {d:+.1f} pp"
        for p, d in zip(probs, deltas)
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text="Decision Threshold (50%)",
        annotation_position="right",
    )
    fig.update_layout(
        title="What-If Scenario Comparison",
        yaxis_title="Churn Probability (%)",
        yaxis_range=[0, min(115, max(probs) * 1.3 + 10)],
        height=400,
    )
    return fig
