# src/hqp/features/__init__.py

"""
This package groups feature-construction modules:
- form.py         : historical form/statistics features (rolling/decayed)
- context.py      : race context encodings (course/going/distance, affinities)
- handicapping.py : draw/weight/official-rating style signals
- build.py        : top-level orchestration and I/O helpers

Design notes
------------
* All feature builders are **leak-free**: they operate on information available
  strictly prior to the current race event.
* Current-race observed fields are namespaced `obs__*` and are **forbidden**
  as inputs (see guards in `build.py`).
"""

from __future__ import annotations

__all__: list[str] = []
