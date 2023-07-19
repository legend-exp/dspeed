from pint import UnitRegistry, set_application_registry

unit_registry = UnitRegistry(auto_reduce_dimensions=True)
set_application_registry(unit_registry)
