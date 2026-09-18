"""
Project 005 — Synthetic Data Generator
Generates customers, products, orders, and multi-channel customer feedback
(reviews, support tickets, survey comments, chat transcripts) with
deliberately embedded ground-truth themes and sentiment so that downstream
NLP (sentiment, topic discovery, emerging-issue detection) can be validated
against known answers.

Scale note: the original spec calls for 1-2M interactions. We generate
120,000 feedback interactions across ~2.5 years, from 8,000 customers and
150 products. This keeps runtime/memory practical while preserving every
statistical pattern (theme mix, emerging issue spike, aspect sentiment,
customer-value/churn linkage) the analytics layer needs to demonstrate.
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_CUSTOMERS = 8000
N_PRODUCTS = 150
N_FEEDBACK = 120000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 6, 30)  # ~2.5 years
TOTAL_DAYS = (END_DATE - START_DATE).days

CHANNELS = ["review", "support_ticket", "survey", "chat"]
CHANNEL_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Beauty", "Sports & Outdoors",
              "Toys & Games", "Office Supplies", "Pet Supplies"]

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"]

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
TIER_WEIGHTS = [0.45, 0.30, 0.18, 0.07]

# ---------------------------------------------------------------------------
# 1. Customers
# ---------------------------------------------------------------------------
customer_ids = [f"CUST-{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
signup_days = np.random.randint(0, TOTAL_DAYS - 30, N_CUSTOMERS)
signup_dates = [START_DATE + timedelta(days=int(d)) for d in signup_days]

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates,
    "region": np.random.choice(REGIONS, N_CUSTOMERS, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
    "tier": np.random.choice(TIERS, N_CUSTOMERS, p=TIER_WEIGHTS),
})
# lifetime value correlated with tier
tier_ltv_base = {"Bronze": 150, "Silver": 450, "Gold": 1200, "Platinum": 3500}
customers["lifetime_value"] = customers["tier"].map(tier_ltv_base) * np.random.lognormal(0, 0.4, N_CUSTOMERS)
customers["lifetime_value"] = customers["lifetime_value"].round(2)
# tier-based churn-risk proxy used for prioritisation/association analysis; no realized churn event is generated
tier_churn_base = {"Bronze": 0.28, "Silver": 0.18, "Gold": 0.10, "Platinum": 0.05}
customers["base_churn_prob"] = customers["tier"].map(tier_churn_base)

# ---------------------------------------------------------------------------
# 2. Products
# ---------------------------------------------------------------------------
product_ids = [f"PROD-{i:04d}" for i in range(1, N_PRODUCTS + 1)]
products = pd.DataFrame({
    "product_id": product_ids,
    "category": np.random.choice(CATEGORIES, N_PRODUCTS),
    "price": np.round(np.random.lognormal(3.2, 0.9, N_PRODUCTS), 2),
})
# a handful of products are deliberately "problem products" with elevated
# quality-complaint rates, so Power BI can surface "worsening products"
products["quality_risk"] = np.random.choice(
    ["normal", "elevated", "high"], N_PRODUCTS, p=[0.80, 0.15, 0.05]
)

# ---------------------------------------------------------------------------
# 3. Orders (backbone connecting customers, products, and feedback)
# ---------------------------------------------------------------------------
N_ORDERS = int(N_FEEDBACK * 1.4)
order_ids = [f"ORD-{i:07d}" for i in range(1, N_ORDERS + 1)]
order_customer = np.random.choice(customer_ids, N_ORDERS)
order_product = np.random.choice(product_ids, N_ORDERS)
order_days = np.random.randint(0, TOTAL_DAYS, N_ORDERS)
order_dates = [START_DATE + timedelta(days=int(d)) for d in order_days]

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": order_customer,
    "product_id": order_product,
    "order_date": order_dates,
})
orders = orders.merge(products[["product_id", "price"]], on="product_id", how="left")
orders["order_value"] = (orders["price"] * np.random.randint(1, 4, N_ORDERS)).round(2)
orders.drop(columns=["price"], inplace=True)

print(f"Generated {len(customers)} customers, {len(products)} products, {len(orders)} orders")

customers.to_csv("data/customers.csv", index=False)
products.to_csv("data/products.csv", index=False)
orders.to_csv("data/orders.csv", index=False)
