"""core — 市场无关的共享库（2026-07 重构批次2 建立）。

- core.data_io — 数据加载 / 时间工具。
  ⚠️ 前向冻结约束：`common_runtime` 与 `history_time_utils` 的真身
  必须留在 `scripts/`（被前向系统 import 闭包引用，import 环境一个
  字符都不能变）；core 侧只放薄 re-export 代理。
- core.db — 数据库初始化 / 环境体检。
"""
