"""Retention ROI Calculator: cost of intervention vs. revenue saved."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.app.clv import EXPECTED_TOTAL_MONTHS


def compute_roi(
    baseline_prob: float,
    new_prob: float,
    monthly_charges: float,
    tenure: float,
    contract: str,
    cost_per_customer: float,
) -> dict:
    """
    Returns ROI metrics for a single intervention.

    Revenue Saved = MonthlyCharges × (baseline_prob − new_prob) × remaining_months
    Net ROI       = Revenue Saved − cost_per_customer
    ROI %         = Net ROI / cost_per_customer × 100  (if cost > 0)
    Break-even    = cost / (MonthlyCharges × prob_reduction)   [months]
    """
    remaining = max(0.0, EXPECTED_TOTAL_MONTHS.get(contract, 24) - tenure)
    prob_reduction = max(0.0, baseline_prob - new_prob)
    revenue_saved = monthly_charges * prob_reduction * remaining
    net_roi = revenue_saved - cost_per_customer
    roi_pct = (net_roi / cost_per_customer * 100) if cost_per_customer > 0 else 0.0
    monthly_revenue_at_risk = monthly_charges * baseline_prob
    break_even_months = (
        cost_per_customer / (monthly_charges * prob_reduction)
        if prob_reduction > 0 and monthly_charges > 0
        else float("inf")
    )
    return {
        "Probability Reduction": prob_reduction,
        "Revenue Saved ($)": round(revenue_saved, 2),
        "Intervention Cost ($)": round(cost_per_customer, 2),
        "Net ROI ($)": round(net_roi, 2),
        "ROI (%)": round(roi_pct, 1),
        "Break-even (months)": round(break_even_months, 1) if break_even_months != float("inf") else "N/A",
        "Monthly Revenue at Risk ($)": round(monthly_revenue_at_risk, 2),
    }


def get_roi_comparison_fig(recs_with_roi: list) -> go.Figure:
    """
    Dual-bar chart: Revenue Saved vs Intervention Cost per action, with Net ROI line.
    recs_with_roi: list of dicts with keys title, revenue_saved, cost, net_roi, roi_pct
    """
    if not recs_with_roi:
        return None

    labels = [r["title"] for r in recs_with_roi]
    revenue = [r["revenue_saved"] for r in recs_with_roi]
    costs = [r["cost"] for r in recs_with_roi]
    net_rois = [r["net_roi"] for r in recs_with_roi]
    roi_pcts = [r["roi_pct"] for r in recs_with_roi]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Revenue Saved vs. Intervention Cost ($)", "Net ROI (%)"),
        column_widths=[0.6, 0.4],
    )

    fig.add_trace(
        go.Bar(name="Revenue Saved", x=labels, y=revenue,
               marker_color="#2ecc71",
               text=[f"${v:,.0f}" for v in revenue],
               textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="Intervention Cost", x=labels, y=costs,
               marker_color="#e74c3c",
               text=[f"${v:,.0f}" for v in costs],
               textposition="outside"),
        row=1, col=1,
    )

    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in roi_pcts]
    fig.add_trace(
        go.Bar(name="ROI %", x=labels, y=roi_pcts,
               marker_color=bar_colors,
               text=[f"{v:.0f}%" for v in roi_pcts],
               textposition="outside",
               showlegend=False),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(
        title="Retention Intervention ROI Analysis",
        barmode="group",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def get_breakeven_fig(
    baseline_prob: float,
    new_prob: float,
    monthly_charges: float,
    cost: float,
    max_months: int = 48,
) -> go.Figure:
    """Line chart showing cumulative revenue saved vs cost over time (break-even view)."""
    months = list(range(0, max_months + 1))
    prob_reduction = max(0.0, baseline_prob - new_prob)
    cumulative_saved = [monthly_charges * prob_reduction * m for m in months]
    cost_line = [cost] * len(months)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=cumulative_saved,
        mode="lines", name="Cumulative Revenue Saved",
        line=dict(color="#2ecc71", width=2.5),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=cost_line,
        mode="lines", name="Intervention Cost",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))

    # Mark break-even
    if prob_reduction > 0 and monthly_charges > 0:
        be = cost / (monthly_charges * prob_reduction)
        if be <= max_months:
            fig.add_vline(
                x=be, line_dash="dot", line_color="orange",
                annotation_text=f"Break-even: {be:.1f} mo",
                annotation_position="top right",
            )

    fig.update_layout(
        title="Break-even Analysis: When Does This Intervention Pay Off?",
        xaxis_title="Months",
        yaxis_title="Cumulative Value ($)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig
