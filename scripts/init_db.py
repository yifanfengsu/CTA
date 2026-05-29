#!/usr/bin/env python3
"""Initialize vnpy database, then import data from extracted DB via sqlite3 ATTACH."""

import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / ".vntrader" / "database.db")
SRC_DB = "/tmp/vnpy_src.db"  # will be created from the uploaded file

# STEP 1: Let peewee/vnpy create the tables in a fresh database
print("Step 1: Creating database schema via vnpy...")
# Delete old DB
for f in Path(DB_PATH).parent.glob("database.db*"):
    f.unlink()

# Import vnpy to trigger table creation
from vnpy.trader.setting import SETTINGS
SETTINGS["database.name"] = "sqlite"
SETTINGS["database.database"] = DB_PATH

from vnpy.trader.database import get_database
db = get_database()
print(f"  Database created at {DB_PATH}")

# IMPORTANT: close peewee connection before using raw sqlite3
db.db.close()
del db

# STEP 2: Import data from source DB
if not os.path.exists(SRC_DB):
    print(f"ERROR: source DB not found at {SRC_DB}")
    sys.exit(1)

print(f"Step 2: Importing data from {SRC_DB}...")
conn = sqlite3.connect(DB_PATH)
conn.execute("ATTACH DATABASE ? AS src", (SRC_DB,))

# Check tables in both
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
src_tables = conn.execute("SELECT name FROM src.sqlite_master WHERE type='table'").fetchall()
print(f"  dst tables: {[t[0] for t in tables]}")
print(f"  src tables: {[t[0] for t in src_tables]}")

# Get column names from dbbardata
cols = [r[1] for r in conn.execute("PRAGMA table_info(dbbardata)").fetchall()]
col_str = ", ".join(cols)
print(f"  dbbardata columns: {cols}")

# Copy dbbardata — symbol should NOT have .GLOBAL (vnpy stores exchange separately)
result = conn.execute(
    f"INSERT INTO dbbardata ({col_str}) "
    f"SELECT id, symbol, exchange, datetime, interval, "
    f"volume, turnover, open_interest, open_price, high_price, low_price, close_price "
    f"FROM src.dbbardata"
)
print(f"  Copied {result.rowcount} rows to dbbardata")

# Copy dbbaroverview
ov_cols = [r[1] for r in conn.execute("PRAGMA table_info(dbbaroverview)").fetchall()]
ov_col_str = ", ".join(ov_cols)
result = conn.execute(f"INSERT INTO dbbaroverview ({ov_col_str}) SELECT {ov_col_str} FROM src.dbbaroverview")
print(f"  Copied {result.rowcount} rows to dbbaroverview")

conn.commit()
try:
    conn.execute("DETACH DATABASE src")
except Exception:
    pass
conn.close()

# Verify
conn2 = sqlite3.connect(DB_PATH)
cnt = conn2.execute("SELECT COUNT(*) FROM dbbardata").fetchone()[0]
print(f"\nStep 3: Verification - {cnt} rows in dbbardata")
conn2.execute("VACUUM")
conn2.close()

size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
print(f"  File size: {size_mb:.1f} MB")
print("Done!")
