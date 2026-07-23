#!/usr/bin/env python3
"""prior-registry skill — 死因/基因登记的轻量查询工具（可独立 import）。

数据源：同目录 registry/deaths.yaml + registry/genomes.yaml（每条来自本仓库真实
研究并标报告路径，数字已逐条回报告核对；无法核实的标 verified: 待核实）。

用途（PIPELINE 第 -1 站 / 第 1 站 / 第 8 站）：
  - query_death(kw)     开题前查'我要赌的机制在本仓库历史上是否被测过、什么级别的死亡'
  - query_genome(kw)    查机制基因（活/死机制的 core/regime/实证）
  - check_conflict(m)   判断新想法是否与 Level 1（机制失败）条目冲突 → 须走 Challenge Path
  - list_reopenable()   列出所有 Level 2 条目及其'跨墙所需条件'，供条件变化时复查

依赖：pyyaml + 标准库。冒烟测试：`python3 registry_query.py`（命中/不误报/冲突判断）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REGISTRY_DIR = Path(__file__).resolve().parent / "registry"

# Challenge Path：与 Level 1（机制失败）条目冲突的想法不自动过滤，须回答二者之一。
CHALLENGE_PATH_QUESTIONS = (
    "① 为什么历史结论这次可能失效？（须给市场结构变化的具体证据，不是'也许不一样'）",
    "② 我的机制如何不同于已死机制？（须说清与命中条目的机制差异，如 order-flow "
    "exhaustion 赌冲击消散而非价格回归）",
)
_CHALLENGE_PATH_RULE = (
    "答得出二者之一 → 走 Stage -1 快速死亡测试；两个都答不出 → 在第 0 站过滤，零成本。"
)

_TOKEN_SPLIT = re.compile(r"[\s\-,_/、，。;；:：()（）]+")


def _load(name: str) -> list[dict[str, Any]]:
    path = _REGISTRY_DIR / name
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    key = "deaths" if "deaths" in data else "genomes"
    return list(data.get(key, []))


def load_deaths() -> list[dict[str, Any]]:
    return _load("deaths.yaml")


def load_genomes() -> list[dict[str, Any]]:
    return _load("genomes.yaml")


def _haystack(entry: dict[str, Any]) -> str:
    """把一条记录里所有可匹配文本拼成小写检索串。"""
    parts: list[str] = list(entry.get("keywords", []) or [])
    for field in ("line", "mechanism", "mechanism_name", "code", "core"):
        val = entry.get(field)
        if val:
            parts.append(str(val))
    return " ".join(parts).lower()


def _match(entry: dict[str, Any], query: str) -> bool:
    """双向匹配：整串子串命中，或任一 ≥2 长度 token 命中检索串。

    既能命中中文整词（'均值回归' ⊂ 检索串），也能命中带空格/连字符的英文
    （'vol targeting' → token 'targeting' ⊂ 'vol-targeting'）。
    """
    q = query.lower().strip()
    if not q:
        return False
    hay = _haystack(entry)
    if q in hay:
        return True
    for tok in _TOKEN_SPLIT.split(q):
        if len(tok) >= 2 and tok in hay:
            return True
    return False


def query_death(mechanism_keyword: str) -> list[dict[str, Any]]:
    """返回命中的死亡条目（含 level 与 next_action）。按 level 升序稳定排序。"""
    hits = [d for d in load_deaths() if _match(d, mechanism_keyword)]
    return sorted(hits, key=lambda d: d.get("level", 9))


def query_genome(mechanism_keyword: str) -> list[dict[str, Any]]:
    """返回命中的机制基因。"""
    return [g for g in load_genomes() if _match(g, mechanism_keyword)]


def check_conflict(proposed_mechanism: str) -> dict[str, Any]:
    """判断新想法是否与 Level 1（机制失败，永久性）条目冲突。

    冲突 → conflict=True + 命中条目 + Challenge Path 须回答的问题。
    不冲突 → conflict=False（想法与地图的 Level 1 结论相容，可直接进 Stage -1）。
    注意：只对 Level 1 判冲突——Level 2 是条件性（见 list_reopenable），Level 3 是
    研究者自己的实现缺陷，二者都不构成对新想法的机制否决。
    """
    level1_hits = [
        d for d in load_deaths()
        if d.get("level") == 1 and _match(d, proposed_mechanism)
    ]
    if not level1_hits:
        return {
            "conflict": False,
            "matched": [],
            "message": "未命中任何 Level 1 机制失败条目——与地图的机制层结论相容，"
                       "可进 Stage -1 快速死亡测试。",
        }
    return {
        "conflict": True,
        "matched": [
            {"line": d["line"], "mechanism": d.get("mechanism"),
             "cause": d.get("cause"), "report": d.get("report")}
            for d in level1_hits
        ],
        "verdict": "须走 Challenge Path",
        "questions": list(CHALLENGE_PATH_QUESTIONS),
        "rule": _CHALLENGE_PATH_RULE,
    }


def list_reopenable() -> list[dict[str, Any]]:
    """列出所有 Level 2（市场约束，条件性）条目及其'跨墙所需条件'。

    条件变化时（如拿到 maker 执行 / 付费数据 / 更低延迟 / 更大资本）逐条复查
    是否可重开。待核实条目（未实测/外部仓库）一并列出并标注。
    """
    out = []
    for d in load_deaths():
        if d.get("level") != 2:
            continue
        out.append({
            "line": d["line"],
            "mechanism": d.get("mechanism"),
            "wall_condition": d.get("wall_condition"),
            "report": d.get("report"),
            "verified": d.get("verified", "已核实"),
        })
    return out


# ─────────────────────────── 冒烟自检 ───────────────────────────

def _smoke() -> int:
    checks: list[tuple[str, bool]] = []

    def chk(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    deaths, genomes = load_deaths(), load_genomes()
    lines = {d["line"] for d in deaths}

    # 结构完整性：每条死因带 report；每条 Level 2 带 wall_condition。
    chk("每条死因带 report", all(d.get("report") for d in deaths))
    chk("每条 Level 2 带 wall_condition",
        all(d.get("wall_condition") for d in deaths if d.get("level") == 2))
    chk("三级齐全", {d.get("level") for d in deaths} >= {1, 2, 3})

    # 命中：已知关键词命中正确条目。
    d_mr = {d["line"] for d in query_death("均值回归")}
    chk("'均值回归' 命中 mr_timescale", "mr_timescale" in d_mr)
    d_vrp = {d["line"] for d in query_death("VRP")}
    chk("'VRP' 命中 ETH+BTC 两条", {"vrp_atm_eth", "vrp_atm_btc"} <= d_vrp)
    chk("'vol targeting'(带空格) 命中 vol_targeting",
        "vol_targeting" in {d["line"] for d in query_death("vol targeting")})
    chk("'order flow exhaustion' 命中 order_flow_exhaustion",
        "order_flow_exhaustion" in {d["line"] for d in query_death("order flow exhaustion")})
    chk("'basis' 命中 basis_arbitrage",
        "basis_arbitrage" in {d["line"] for d in query_death("basis")})

    # 基因命中。
    chk("'趋势' 命中 B2_4h 基因",
        any(g.get("code") == "B2_4h" for g in query_genome("趋势")))

    # 不误报：不存在的关键词返回空。
    chk("不存在关键词不误报(death)", query_death("zk-rollup-mev-quantum") == [])
    chk("不存在关键词不误报(genome)", query_genome("zk-rollup-mev-quantum") == [])

    # 冲突判断：已知冲突（Level 1）与不冲突（趋势=地图允许方向）。
    conf = check_conflict("mean reversion fade the extreme")
    chk("冲突案例判 conflict=True", conf["conflict"] is True)
    chk("冲突案例命中 Level 1 条目", len(conf["matched"]) >= 1)
    chk("冲突案例给 Challenge Path 问题", len(conf.get("questions", [])) == 2)
    noconf = check_conflict("trend-following continuation harvesting")
    chk("不冲突案例判 conflict=False（趋势=地图允许）", noconf["conflict"] is False)
    conf_of = check_conflict("order-flow exhaustion rebound")
    chk("order-flow exhaustion 判冲突（Level 1）", conf_of["conflict"] is True)

    # 可重开列表：全 Level 2，各带 wall_condition，含已知条目。
    reop = list_reopenable()
    chk("list_reopenable 非空", len(reop) > 0)
    chk("list_reopenable 各带 wall_condition", all(r.get("wall_condition") for r in reop))
    chk("list_reopenable 含 breakout_pullback",
        "breakout_pullback" in {r["line"] for r in reop})

    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nregistry_query 冒烟：{n_pass}/{len(checks)} PASS "
          f"(deaths={len(deaths)}, genomes={len(genomes)}, "
          f"L1={sum(d.get('level') == 1 for d in deaths)}, "
          f"L2={sum(d.get('level') == 2 for d in deaths)}, "
          f"L3={sum(d.get('level') == 3 for d in deaths)})")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
