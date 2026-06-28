# Institutional Data Cost Survey — the data threshold for *true* large-scale factors

**Pure reconnaissance, nothing purchased (2026-06-28).** This documents what it would
cost to buy the data that free Binance Vision *cannot* give us: **survivorship-free,
delisted-inclusive, ideally point-in-time** history across hundreds of CEX perps/spot.
That capability is the precondition for a *real* large-scale cross-sectional factor
program (free data is survivor-biased → alpha optimistic; see README §Q1/Q6).

## Why free data is not enough (the threshold being priced)
Two distinct gaps, increasing in cost:
1. **Survivorship-free raw history** — includes coins that died (LUNA/FTT/etc.),
   expired, or were renamed. Removes the optimistic bias of "today's listed coins".
2. **Point-in-time (PIT) universe membership** — *what was tradeable / in-index on each
   historical date*, so universe selection itself has no look-ahead. Strictly harder
   than (1): you either reconstruct it from raw delisted data, or buy a curated index
   with PIT constituents.

## Vendor reconnaissance (list prices / public ranges)

| Vendor | Survivorship-free? | PIT universe? | Indicative cost | Notes |
|---|---|---|---|---|
| **Tardis.dev** | **Yes, explicit** ("survival-bias-free … includes delisted, expired, renamed") | Self-reconstruct from raw | **Min order $300**; invoicing kicks in **>$6,000**; plan = exchange-group (Perps/Options/Spot/Derivatives/All) × interval (mo/qtr/yr) | Cheapest credible survivor-free source. Tick + trades + book + funding CSV. A serious multi-year, multi-exchange perp pull realistically **~$1k–6k one-time**. |
| **CoinAPI (Flat Files)** | Partial (historical archive) | No (raw) | **Usage-based**, ~**$170–465/mo** for *small* scope (top-10 across a few exchanges); scales with GB downloaded | Transparent self-serve; cost explodes with breadth/granularity. Good for narrow pulls, not curated. |
| **Amberdata** | Archive-complete | Vendor indices available | Self-serve by exchange selection (monthly/annual + S3 bulk); **advanced = enterprise quote** | CC checkout for standard; rate-limit tiers public, full pricing not. |
| **Kaiko** | Yes (tick archive is complete by construction) | **Yes** (curated Reference Rates + indices w/ PIT) | **Avg ~$28.5k/yr; range ~$9.5k–$55k/yr** (Vendr transaction data). Product tiers: L1 aggregations **from $1k/mo**, L1 tick **from $1.5k/mo** | Institutional gold standard. 100+ exchanges, 35k+ pairs. Custom enterprise contracts. |
| **Coin Metrics** | Yes | **Yes** (Reference Rates, 1000+ assets; Market Data Pro 400k+ markets) | **No public price** — enterprise quote (peer of Kaiko, expect **~$10k–50k+/yr**) | Curated reference data + indices; institutional negotiation. |

## Bottom line — the price of removing survivor bias

- **Lower bound (raw survivor-free, build PIT yourself): ~$1,000–6,000 one-time** via
  Tardis.dev flat files (all-derivatives, multi-year). This buys gap #1 (delisted
  coins) but you still engineer PIT universe membership yourself.
- **Institutional grade (curated + PIT constituents + reference rates + support):
  ~$10,000–55,000 / year** via Kaiko or Coin Metrics. This buys gaps #1 **and** #2.

So the **data threshold for a genuine survivor-free large-scale factor program is
roughly $1–6k to start (Tardis) and $10–55k/yr to do it institutionally.** That spend
is only justified if this Stage-A free gate first shows alpha *improves with scale* on
the (optimistic) free sample. If the free sample already says "no scale edge", buying
survivor-free data — which makes alpha *worse*, not better — cannot rescue it, and the
spend is saved. (See README §Q5/Q6 for the verdict.)

## Sources
- [Kaiko pricing — Vendr buyer guide](https://www.vendr.com/buyer-guides/kaiko) · [Kaiko pricing & contracts](https://www.kaiko.com/about-kaiko/pricing-and-contracts) · [Kaiko L1/L2 data](https://www.kaiko.com/products/data-feeds/l1-l2-data)
- [Tardis.dev data FAQ (survivorship-bias-free)](https://docs.tardis.dev/faq/data) · [Tardis billing & subscriptions](https://docs.tardis.dev/faq/billing-and-subscriptions) · [Tardis order form](https://tardis.dev/#order)
- [Coin Metrics market data feed](https://coinmetrics.io/market-data-feed/) · [Coin Metrics prices/reference rates](https://coinmetrics.io/prices/)
- [CoinAPI Flat Files pricing](https://www.coinapi.io/products/flat-files/pricing) · [CoinAPI Flat Files overview](https://www.coinapi.io/products/flat-files)
- [Amberdata pricing](https://www.amberdata.io/pricing) · [Amberdata market data](https://www.amberdata.io/market-data)
