"""experiments.factors — DAILY cross-sectional factor research harness.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on market data.

This package pivots away from the (dead) intraday VWAP family (see
experiments/results/NEUTRAL_SUMMARY.md — 0 beta-neutral edges on free IEX intraday
data) to DAILY cross-sectional factor strategies on FULL-QUALITY split+dividend-adjusted
free daily bars. Daily bars are full quality on the Alpaca free tier; the IEX
2-3%-of-volume distortion only affects intraday data.
"""
