"""Model-driven intervention recommender: estimate retention action impact."""
from typing import List, Dict, Callable

INTERVENTIONS = [
    {
        "id": "upgrade_2yr",
        "condition": lambda c: c.get("Contract") == "Month-to-month",
        "change": {"Contract": "Two year"},
        "title": "📋 Upgrade to 2-Year Contract",
        "description": (
            "Two-year contracts have the lowest churn rate across all segments. "
            "Offer an incentive (e.g. free month or discounted first year) to lock in the customer."
        ),
        "effort": "Medium",
        "category": "Contract",
    },
    {
        "id": "upgrade_1yr",
        "condition": lambda c: c.get("Contract") == "Month-to-month",
        "change": {"Contract": "One year"},
        "title": "📋 Upgrade to 1-Year Contract",
        "description": (
            "Even a 1-year commitment significantly reduces churn risk versus month-to-month. "
            "Lower barrier than a 2-year offer."
        ),
        "effort": "Low",
        "category": "Contract",
    },
    {
        "id": "add_tech_support",
        "condition": lambda c: (
            c.get("TechSupport") == "No" and c.get("InternetService") != "No"
        ),
        "change": {"TechSupport": "Yes"},
        "title": "🛠️ Add Tech Support",
        "description": (
            "Customers without tech support who experience issues are more likely to leave. "
            "Bundling support reduces frustration-driven churn."
        ),
        "effort": "Low",
        "category": "Services",
    },
    {
        "id": "add_online_security",
        "condition": lambda c: (
            c.get("OnlineSecurity") == "No" and c.get("InternetService") != "No"
        ),
        "change": {"OnlineSecurity": "Yes"},
        "title": "🔒 Add Online Security",
        "description": (
            "Security add-ons increase perceived service value and create meaningful switching costs."
        ),
        "effort": "Low",
        "category": "Services",
    },
    {
        "id": "add_online_backup",
        "condition": lambda c: (
            c.get("OnlineBackup") == "No" and c.get("InternetService") != "No"
        ),
        "change": {"OnlineBackup": "Yes"},
        "title": "☁️ Add Online Backup",
        "description": (
            "Cloud backup increases stickiness — customers with stored data are less likely to switch providers."
        ),
        "effort": "Low",
        "category": "Services",
    },
    {
        "id": "device_protection",
        "condition": lambda c: (
            c.get("DeviceProtection") == "No" and c.get("InternetService") != "No"
        ),
        "change": {"DeviceProtection": "Yes"},
        "title": "🛡️ Add Device Protection",
        "description": "Device protection plans create additional retention through asset coverage.",
        "effort": "Low",
        "category": "Services",
    },
    {
        "id": "switch_auto_payment",
        "condition": lambda c: c.get("PaymentMethod") == "Electronic check",
        "change": {"PaymentMethod": "Bank transfer (automatic)"},
        "title": "💳 Switch to Automatic Payment",
        "description": (
            "Electronic check customers churn at higher rates. "
            "Auto bank transfer removes monthly payment friction and improves retention."
        ),
        "effort": "Low",
        "category": "Billing",
    },
    {
        "id": "loyalty_discount",
        "condition": lambda c: float(c.get("MonthlyCharges", 0)) > 70,
        "change": {},  # handled specially below
        "title": "💰 Offer 15% Loyalty Discount",
        "description": (
            "High-charge customers are price-sensitive. "
            "A targeted loyalty discount can reduce perceived cost and prevent churn."
        ),
        "effort": "Medium",
        "category": "Pricing",
    },
]


def get_recommendations(customer_data: dict, predict_fn: Callable) -> List[Dict]:
    """
    Evaluate each applicable intervention against the model and return a list
    of recommendation dicts sorted by estimated churn probability reduction.
    """
    try:
        baseline_result = predict_fn(customer_data)
        baseline_prob = baseline_result.get("churn_probability", 0.5)
    except Exception:
        return []

    recs = []
    for interv in INTERVENTIONS:
        try:
            if not interv["condition"](customer_data):
                continue

            modified = customer_data.copy()
            if interv["id"] == "loyalty_discount":
                modified["MonthlyCharges"] = max(
                    float(customer_data.get("MonthlyCharges", 65)) * 0.85, 25.0
                )
            else:
                modified.update(interv["change"])

            result = predict_fn(modified)
            new_prob = result.get("churn_probability", baseline_prob)
            delta = baseline_prob - new_prob

            recs.append({
                **interv,
                "baseline_prob": baseline_prob,
                "new_prob": new_prob,
                "delta": delta,
                "delta_pct": delta * 100,
            })
        except Exception:
            continue

    recs.sort(key=lambda x: x["delta"], reverse=True)
    return recs
