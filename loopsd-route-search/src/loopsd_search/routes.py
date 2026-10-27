"""Paper routes use 1-based inclusive endpoints; execution indices are 0-based."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    start: int | None
    end: int | None
    repeat_count: int = 2

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("Both route endpoints must be specified, or both absent")
        if self.start is not None and (self.start < 1 or self.end < self.start):
            raise ValueError("Expected 1 <= start <= end, inclusive")
        if self.start is not None and self.repeat_count < 2:
            raise ValueError("A loop route must execute its interval at least twice")

    @classmethod
    def parse(cls, value: str) -> "Route":
        if value == "standard":
            return cls(None, None)
        interval, separator, factor = value.partition("x")
        parts = interval.split(":")
        if len(parts) != 2:
            raise ValueError(
                "Route must be 'standard' or 'start:end[xN]' (1-based inclusive)"
            )
        if separator and (not factor.isdigit() or int(factor) < 2):
            raise ValueError("Loop factor suffix must be xN where N >= 2")
        return cls(int(parts[0]), int(parts[1]), int(factor) if separator else 2)

    @property
    def id(self) -> str:
        if self.start is None:
            return "standard"
        suffix = "" if self.repeat_count == 2 else f"x{self.repeat_count}"
        return f"{self.start}:{self.end}{suffix}"

    def indices(self, num_layers: int) -> tuple[int, ...]:
        if num_layers < 1:
            raise ValueError("Model must have at least one layer")
        base = tuple(range(num_layers))
        if self.start is None:
            return base
        if self.end > num_layers:
            raise ValueError(f"Route {self.id} exceeds model depth {num_layers}")
        repeated_block = base[self.start - 1 : self.end] * (self.repeat_count - 1)
        return base[: self.end] + repeated_block + base[self.end :]


def enumerate_routes(
    num_layers: int,
    candidates: list[str] | None,
    loop_factors: list[int] | None = None,
) -> list[Route]:
    if candidates is None:
        factors = loop_factors or [2]
        if any(factor < 2 for factor in factors) or len(set(factors)) != len(factors):
            raise ValueError("loop_factors must contain unique integers >= 2")
        loops = [
            Route(i, j, factor)
            for factor in factors
            for i in range(1, num_layers + 1)
            for j in range(i, num_layers + 1)
        ]
    else:
        loops = [Route.parse(value) for value in candidates]
        if len({route.id for route in loops}) != len(loops):
            raise ValueError("Duplicate candidate routes")
        loops = [route for route in loops if route.start is not None]
        if not loops:
            raise ValueError("Specify at least one nonstandard candidate")
    for route in loops:
        route.indices(num_layers)
    return [Route(None, None), *loops]
