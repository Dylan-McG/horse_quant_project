from __future__ import annotations

import types


def test_packages_importable() -> None:
    import hqp
    import hqp.checks
    import hqp.features
    import hqp.io
    import hqp.market
    import hqp.models
    import hqp.plots
    import hqp.ratings
    import hqp.scripts
    import hqp.split

    modules = [
        hqp,
        hqp.checks,
        hqp.features,
        hqp.io,
        hqp.market,
        hqp.models,
        hqp.plots,
        hqp.ratings,
        hqp.scripts,
        hqp.split,
    ]

    # Touch __name__ to silence "unused import" and assert they're modules
    assert all(isinstance(m, types.ModuleType) and isinstance(m.__name__, str) for m in modules)

    # __version__ must exist on the root package
    assert isinstance(hqp.__version__, str)
