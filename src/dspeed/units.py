import pint

unit_registry = pint.get_application_registry()
# match the short-form default used across the legend-exp ecosystem
# (lgdo.units does the same on import)
unit_registry.formatter.default_format = "~P"
