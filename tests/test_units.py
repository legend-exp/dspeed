import pint


def test_shares_application_registry():
    # Importing dspeed.units must not replace the application registry, and
    # quantities created via the bare pint API must interoperate with dspeed's
    # registry (regression for cross-registry errors in consumer code).
    app_reg_before = pint.get_application_registry()
    from dspeed.units import unit_registry as ureg

    assert pint.get_application_registry() is app_reg_before

    consumer_q = pint.Quantity(1, "us")
    dspeed_q = ureg.Quantity(1000, "ns")
    assert (consumer_q + dspeed_q).to("us").magnitude == 2
