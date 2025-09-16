from __future__ import annotations


def test_packages_importable() -> None:
    import hqp
    import hqp.checks
    import hqp.features
    import hqp.io
    import hqp.market
    import hqp.modeling
    import hqp.plots
    import hqp.ratings
    import hqp.scripts
    import hqp.split

    assert isinstance(hqp.__version__, str)
