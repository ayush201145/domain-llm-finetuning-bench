import json
import os
from typing import List, Dict

def generate_synthetic_sql_dataset() -> List[Dict]:
    """Generates synthetic domain-specific SQL generation instruction pairs"""
    samples = [
        {
            "instruction": "Retrieve top 5 customers with highest Monthly Recurring Revenue (MRR) in 2026.",
            "input": "Table: customers (id, name, mrr, status, created_at)",
            "output": "SELECT id, name, mrr FROM customers WHERE status = 'ACTIVE' AND created_at >= '2026-01-01' ORDER BY mrr DESC LIMIT 5;"
        },
        {
            "instruction": "Calculate 30-day user churn rate per organization tenant.",
            "input": "Table: tenant_activity (tenant_id, active_users, churned_users, snapshot_date)",
            "output": "SELECT tenant_id, SUM(churned_users) * 100.0 / NULLIF(SUM(active_users), 0) AS churn_rate FROM tenant_activity WHERE snapshot_date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY tenant_id;"
        }
    ]
    return samples

def save_dataset(output_path: str = "dataset.json"):
    data = generate_synthetic_sql_dataset()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Curated {len(data)} domain instruction samples saved to {output_path}")

if __name__ == "__main__":
    save_dataset()
