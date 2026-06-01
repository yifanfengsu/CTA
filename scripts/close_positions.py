#!/usr/bin/env python3
"""Check and optionally close all OKX DEMO positions."""
import os, json, requests, datetime, base64, hmac, hashlib, sys
from dotenv import load_dotenv

load_dotenv('/run-project/vnpy_strategy_test/CTA/.env')
REST_URL = "https://www.okx.com"

def sign(ts, method, path, body=""):
    secret = os.getenv("OKX_SECRET_KEY", "")
    sign_str = ts + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()).decode()

def headers(method, path, body=""):
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    return {
        "OK-ACCESS-KEY": os.getenv("OKX_API_KEY", ""),
        "OK-ACCESS-SIGN": sign(ts, method, path, body),
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": os.getenv("OKX_PASSPHRASE", ""),
        "Content-Type": "application/json",
        "x-simulated-trading": "1",
    }

def okx_get(path):
    return requests.get(f"{REST_URL}{path}", headers=headers("GET", path), timeout=10).json()

def okx_post(path, body):
    body_str = json.dumps(body)
    return requests.post(f"{REST_URL}{path}", data=body_str,
                         headers=headers("POST", path, body_str), timeout=10).json()

# Query positions
print("=== Current Positions ===")
resp = okx_get("/api/v5/account/positions?instType=SWAP")
positions = []
for p in resp.get("data", []):
    pos = float(p.get("pos", 0))
    if pos != 0:
        side = "long" if pos > 0 else "short"
        close_side = "sell" if pos > 0 else "buy"
        print(f"  {p['instId']}: {side} pos={pos} avgPx={p.get('avgPx','?')} upl={p.get('upl','?')}")
        positions.append((p["instId"], pos, close_side))

if not positions:
    print("  No open positions")
    sys.exit(0)

# Close positions if --close flag
if "--close" in sys.argv:
    print("\n=== Closing All Positions ===")
    for inst_id, pos, close_side in positions:
        sz = str(int(abs(pos)))
        result = okx_post("/api/v5/trade/order", {
            "instId": inst_id,
            "tdMode": "cross",
            "side": close_side,
            "ordType": "market",
            "sz": sz,
        })
        code = result.get("code", "")
        if code == "0":
            ord_id = result["data"][0].get("ordId", "")[:8]
            print(f"  {inst_id}: MARKET {close_side} sz={sz} → {ord_id} OK")
        else:
            print(f"  {inst_id}: FAILED {result.get('msg','?')}")
    print("Done")
else:
    print("\nRun with --close to close all positions")
