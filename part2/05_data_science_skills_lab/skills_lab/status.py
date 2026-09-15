"""S08 - service status: what this lab is, what it can compute, and what data is ready.

Run it directly:  python -m skills_lab.status
"""

from __future__ import annotations

import json

from skills_lab import catalog
from skills_lab.config import CACHE_FILES
from skills_lab.data.acquire import is_prepared


def status() -> dict:
    datasets = {key: ("ready" if is_prepared(key) else "missing") for key in CACHE_FILES}
    return {
        "service": "Data Science Skills Mastery Lab",
        "skills_in_catalog": catalog.skill_count(),
        "categories": catalog.CATEGORIES,
        "benchmarks": [
            catalog.WORKSPACE_LABELS[view]
            for view in catalog.View
            if view is not catalog.View.CATALOG
        ],
        "datasets": datasets,
        "ready": all(state == "ready" for state in datasets.values()),
        "prepare_command": "python scripts/prepare_data.py",
    }


def main() -> None:
    print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
