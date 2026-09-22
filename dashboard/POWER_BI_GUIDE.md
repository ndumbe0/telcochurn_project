# Power BI Dashboard — Telco Churn Analyzer

Built from `telco_churn_unified.csv` in this folder (5,042 customers,
26.5% churn) and `telco_churn_summary.csv` (pre-aggregated churn rates per
dimension). ~10 minutes in Power BI Desktop (free).

## 1. Load the data

1. **Get data → Text/CSV** → `telco_churn_unified.csv` → **Transform Data**.
2. Set types: `tenure`, `tenure_group` (Text), `churned` → Whole Number,
   `senior_citizen` → Whole Number (or True/False),
   `monthly_charges`, `total_charges` → Decimal Number, rest → Text.
3. Rename the query in the Queries pane to `Telco`. Close & Apply.
4. (Optional) Also load `telco_churn_summary.csv` as `TelcoSummary` for fast
   dimension cards.

## 2. DAX measures (Modeling → New Measure)

```dax
Churn Rate = AVERAGE(Telco[churned])

Customers = COUNTROWS(Telco)

Churned Customers = SUM(Telco[churned])

Avg Monthly Charges = AVERAGE(Telco[monthly_charges])

Revenue at Risk USD =
SUMX(FILTER(Telco, Telco[churned] = 1), Telco[monthly_charges])

Churn Rate — Month-to-month vs Year =
DIVIDE(
    CALCULATE([Churn Rate], Telco[contract] = "Month-to-month"),
    CALCULATE([Churn Rate], Telco[contract] = "One year"))
```

## 3. Pages and visuals

### Page 1 — Churn Overview
| Visual | Fields |
|---|---|
| **Card** | `Churn Rate` (format as %) |
| **Card** | `Customers` |
| **Card** | `Revenue at Risk USD` |
| **Donut** | Legend: `churn` • Values: `Customers` |
| **Clustered bar** | Axis: `contract` • Values: `Churn Rate` |
| **Line/area** | Axis: `tenure_group` (order 0-12…61-72) • Values: `Churn Rate` |

### Page 2 — Drivers
| Visual | Fields |
|---|---|
| **Bar chart** | Axis: `payment_method` • Values: `Churn Rate` (sort desc, Top 5) |
| **Clustered bar** | Axis: `internet_service` • Legend: `online_security` • Values: `Churn Rate` |
| **Matrix** | Rows: `gender`, `senior_citizen`, `partner`, `dependents` • Values: `Churn Rate`, `Customers` + color-scale conditional formatting |
| **Scatter** | X: `avg tenure` Y: `avg monthly charges` Details: `contract` • filter via `TelcoSummary` if loaded |

### Page 3 — Retention levers
| Visual | Fields |
|---|---|
| **Bar** | Axis: add-on columns one by one (`tech_support`, `online_backup`, `device_protection`) via a what-if parameter, or bind `TelcoSummary` rows where dimension = those columns |
| **Slicer** | `contract` (Sync across all pages) |
| **KPI** | Target: `Churn Rate` = 20% • Value: current |

Tip: `churned` (0/1) exists precisely so AVERAGE gives you churn rate without
filters — use it on every visual.

## 4. Optional: SQL Server

Run `lp2_telco_churn.sql` in SSMS → database `LP2_TelcoChurn`, table
`telco_churn` + views `v_churn_by_contract`, `v_churn_by_payment`,
`v_churn_by_tenure_group`, `v_churn_by_internet`, `v_senior_dependents`,
`v_revenue_at_risk`. Bind visuals directly to the views to skip DAX.

## 5. Tableau

Open `lp2_telco_dashboard.twb` from this folder (it expects
`telco_churn_unified.csv` alongside it). 5 sheets: Churn Rate by Contract,
Churn by Tenure Group, Churn by Payment Method, Churn by Internet Service,
Monthly Revenue by Churn.
