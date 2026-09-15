"""Clearly-labelled illustrative inputs.

Only one thing in this project is not derived from real data: the checkout funnel. Online
Retail II records completed orders, not page-level events, and no comparable free event
stream exists, so the funnel stages below are a teaching example. Every surface that shows
them must carry the label in ``FUNNEL_LABEL``.
"""

from __future__ import annotations

FUNNEL_LABEL = "Illustrative example - not measured from an event stream"

FUNNEL_STAGES = [
    {"stage": "Homepage visit", "users": 100_000},
    {"stage": "Product page view", "users": 62_400},
    {"stage": "Add to cart", "users": 28_100},
    {"stage": "Checkout started", "users": 17_900},
    {"stage": "Purchase completed", "users": 11_200},
]
