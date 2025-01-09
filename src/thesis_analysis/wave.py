import re
from dataclasses import dataclass

import laddu as ld

__all__ = ['Wave']


@dataclass
class Wave:
    l: int  # noqa: E741
    m: int
    r: str

    @property
    def plot_index(self, *, double: bool = False) -> tuple[int, int]:
        if double:
            if self.l == 0:
                return (0, 0)
            else:
                return (1, 0)
        if self.m == 0:
            if self.l == 0:
                return (0, 0)
            else:
                return (1, 0)
        if self.m > 0:
            return (0, self.m)
        else:
            return (1, abs(self.m))

    @property
    def positive(self) -> bool:
        return self.r == '+'

    @property
    def negative(self) -> bool:
        return self.r == '-'

    @staticmethod
    def from_string(string: str) -> 'Wave':
        m = re.match(r'^(\d)([+-]?\d)([+-])$', string)
        if not m:
            raise Exception(f'The string {string} is not a valid wave!')
        return Wave(int(m.group(1)), int(m.group(2)), m.group(3))

    @property
    def amplitude_name(self) -> str:
        return f"z{self.l}{'+' if self.m >= 0 else '-'}{abs(self.m)}{self.r}"

    @property
    def coefficient_name(self) -> str:
        return f"c{self.l}{'+' if self.m >= 0 else '-'}{abs(self.m)}{self.r}"

    @property
    def letter(self) -> str:
        match self.l:
            case 0:
                return 'S'
            case 1:
                return 'P'
            case 2:
                return 'D'
            case 3:
                return 'F'
            case _:
                return chr(ord('G') + self.l - 4)

    @property
    def latex(self) -> str:
        return f"${self.letter}_{{{'+' if self.m >= 0 else '-'}{abs(self.m)}}}^{{({self.r})}}$"

    @property
    def latex_group(self) -> str:
        return f"${self.letter}_{{{'+' if self.m >= 0 else '-'}{abs(self.m)}}}$"

    @property
    def amplitude(self) -> ld.amplitudes.Amplitude:
        angles = ld.Angles(0, [1], [2], [2, 3])
        polarization = ld.Polarization(0, [1])
        return ld.Zlm(
            self.amplitude_name,
            self.l,  # type: ignore
            self.m,  # type: ignore
            self.r,  # type: ignore
            angles,
            polarization,
        )

    @property
    def coefficient(self, anchor: bool = False) -> ld.amplitudes.Amplitude:
        if anchor:
            return ld.Scalar(
                self.coefficient_name,
                ld.parameter(f'{self.coefficient_name} real'),
            )
        else:
            return ld.ComplexScalar(
                self.coefficient_name,
                ld.parameter(f'{self.coefficient_name} real'),
                ld.parameter(f'{self.coefficient_name} imag'),
            )
