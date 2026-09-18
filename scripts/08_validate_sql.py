import duckdb
import re

con = duckdb.connect()
con.execute("CREATE TABLE customers AS SELECT * FROM read_csv_auto('data/customers.csv')")
con.execute("CREATE TABLE products AS SELECT * FROM read_csv_auto('data/products.csv')")
con.execute("CREATE TABLE orders AS SELECT * FROM read_csv_auto('data/orders.csv')")
con.execute("CREATE TABLE feedback_final_scored AS SELECT * FROM read_csv_auto('data/feedback_final_scored.csv')")
con.execute("CREATE TABLE daily_theme_trends AS SELECT * FROM read_csv_auto('data/daily_theme_trends.csv')")

with open("sql/analytics_queries.sql") as f:
    sql_text = f.read()

# split on statement-ending semicolons that are followed by a comment block or EOF
statements = [s.strip() for s in sql_text.split(";\n\n") if s.strip()]
# clean trailing standalone semicolons
statements = [s.rstrip(";").strip() for s in statements]

for i, stmt in enumerate(statements, 1):
    # skip the parameterized customer-360 query (needs an app-layer param)
    if ":customer_id" in stmt:
        stmt_test = stmt.replace(":customer_id", "'CUST-000001'")
    else:
        stmt_test = stmt
    try:
        result = con.execute(stmt_test).fetchdf()
        print(f"Query {i}: OK — {len(result)} rows returned")
    except Exception as e:
        print(f"Query {i}: FAILED — {e}")
        print(stmt_test[:300])
        print("---")
