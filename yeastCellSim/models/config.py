from dataclasses import dataclass


@dataclass
class CellProperties:
    radius: list


@dataclass
class SimulationConfig:
    bound_volume: float
    bud_angle_pattern: str
    prune_collisions: bool


@dataclass
class Configuration:
    cell_properties: CellProperties
    simulation: SimulationConfig
    generations: int
    root: list
    verbose: bool
