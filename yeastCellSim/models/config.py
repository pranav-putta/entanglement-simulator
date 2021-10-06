from dataclasses import dataclass


@dataclass
class CellProperties:
    radius: list


@dataclass
class Configuration:
    cell_properties: CellProperties
    generations: int
    root: list
