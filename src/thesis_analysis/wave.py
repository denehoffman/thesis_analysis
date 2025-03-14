import re
from itertools import combinations

import laddu as ld

from thesis_analysis import colors


class Wave:
    def __init__(self, l: int, m: int, r: str) -> None:  # noqa: E741
        assert l in {0, 1, 2, 3}, 'L < 0 or L > 3 is not supported!'
        assert -l <= m <= l
        assert r in {'-', '+'}
        self.l: int = l  # noqa: E741
        self.m: int = m
        self.r: str = r

    def __lt__(self, other: 'Wave') -> bool:
        return (self.l, self.m, self.r) < (other.l, other.m, other.r)

    def encode(self) -> int:
        return (self.l << 4) | ((self.m + 3) << 1) | (1 if self.positive else 0)

    @staticmethod
    def decode(code: int) -> 'Wave':
        return Wave(
            (code >> 4) & 3, ((code >> 1) & 7) - 3, '+' if code & 1 else '-'
        )

    @staticmethod
    def encode_waves(waves: set['Wave']) -> int:
        waves = set(waves)
        encoded = 0
        for wave in waves:
            encoded = (encoded << 6) | wave.encode()
        return encoded

    @staticmethod
    def decode_waves(encoded: int) -> set['Wave']:
        waves = []
        while encoded:
            wave_data = encoded & 0b111111
            waves.append(Wave.decode(wave_data))
            encoded >>= 6
        return set(waves)

    @property
    def plot_color(self) -> str:
        if self.positive:
            return colors.red
        else:
            return colors.blue

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
    def zlm_name(self) -> str:
        return f'z{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def coefficient_name(self) -> str:
        return f'c{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def kmatrix_names(self) -> list[str]:
        return [self.kmatrix_name_f, self.kmatrix_name_a]

    @property
    def kmatrix_name_f(self) -> str:
        return f'f{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def kmatrix_name_a(self) -> str:
        return f'a{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

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
                raise Exception('L > 3 is not supported!')

    @staticmethod
    def power_set(waves: set['Wave']) -> list[set['Wave']]:
        return [
            set(subset)
            for r in range(1, len(waves) + 1)
            for subset in combinations(waves, r)
        ]

    @staticmethod
    def get_amplitude_names(wave: 'Wave', *, mass_dependent: bool) -> list[str]:
        names: list[str] = []
        names.append(wave.zlm_name)
        if mass_dependent:
            names.extend(wave.kmatrix_names)
        else:
            names.append(wave.coefficient_name)
        return names

    @staticmethod
    def get_waveset_names(
        waveset: set['Wave'], *, mass_dependent: bool
    ) -> list[str]:
        return list(
            set(
                name
                for wave in waveset
                for name in Wave.get_amplitude_names(
                    wave, mass_dependent=mass_dependent
                )
            )
        )

    @property
    def latex(self) -> str:
        return f'${self.letter}_{{{"+" if self.m >= 0 else "-"}{abs(self.m)}}}^{{({self.r})}}$'

    @property
    def latex_group(self) -> str:
        return f'${self.letter}_{{{"+" if self.m >= 0 else "-"}{abs(self.m)}}}$'

    def zlm(self, manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        angles = ld.Angles(0, [1], [2], [2, 3])
        polarization = ld.Polarization(0, [1], 0)
        return manager.register(
            ld.Zlm(
                self.zlm_name,
                self.l,
                self.m,
                self.r,
                angles,
                polarization,
            )
        )

    def coefficient(self, manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        if self.l == 0:
            return manager.register(
                ld.Scalar(
                    self.coefficient_name,
                    ld.parameter(f'{self.coefficient_name} real'),
                )
            )
        else:
            return manager.register(
                ld.ComplexScalar(
                    self.coefficient_name,
                    ld.parameter(f'{self.coefficient_name} real'),
                    ld.parameter(f'{self.coefficient_name} imag'),
                )
            )

    def kmatrix(self, manager: ld.Manager) -> list[ld.amplitudes.AmplitudeID]:
        res_mass = ld.Mass([2, 3])
        wave_subinfo = f'{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'
        if self.l == 0:
            f = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixF0(
                    self.kmatrix_name_f,
                    (
                        (ld.constant(0), ld.constant(0)),
                        (
                            ld.parameter(f'f0(980)({wave_subinfo}) real'),
                            ld.constant(0),
                        ),
                        (
                            ld.parameter(f'f0(1370)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1370)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f0(1500)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1500)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f0(1710)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1710)({wave_subinfo}) imag'),
                        ),
                    ),
                    2,
                    res_mass,
                )
            )
            a = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixA0(
                    self.kmatrix_name_a,
                    (
                        (
                            ld.parameter(f'a0(980)({wave_subinfo}) real'),
                            ld.parameter(f'a0(980)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'a0(1450)({wave_subinfo}) real'),
                            ld.parameter(f'a0(1450)({wave_subinfo}) imag'),
                        ),
                    ),
                    1,
                    res_mass,
                )
            )
        elif self.l == 2:
            f = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixF2(
                    self.kmatrix_name_f,
                    (
                        (
                            ld.parameter(f'f2(1270)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1270)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1525)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1525)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1810)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1810)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1950)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1950)({wave_subinfo}) imag'),
                        ),
                    ),
                    2,
                    res_mass,
                )
            )
            a = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixA2(
                    self.kmatrix_name_a,
                    (
                        (
                            ld.parameter(f'a2(1320)({wave_subinfo}) real'),
                            ld.parameter(f'a2(1320)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'a2(1700)({wave_subinfo}) real'),
                            ld.parameter(f'a2(1700)({wave_subinfo}) imag'),
                        ),
                    ),
                    1,
                    res_mass,
                )
            )
        else:
            raise Exception('K-Matrix for L != 0, 2 is not supported!')
        return [f, a]

    @staticmethod
    def phase_space_factor(manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        m_resonance = ld.Mass([2, 3])
        m_1 = ld.Mass([2])
        m_2 = ld.Mass([3])
        m_recoil = ld.Mass([1])
        s = ld.Mandelstam([0], [], [2, 3], [1], channel='s')
        return manager.register(
            ld.PhaseSpaceFactor('kappa', m_recoil, m_1, m_2, m_resonance, s)
        )

    @staticmethod
    def get_model(
        waves: set['Wave'], *, mass_dependent: bool, phase_factor: bool = False
    ) -> ld.Model:
        pos_r_waves = [wave for wave in waves if wave.positive]
        neg_r_waves = [wave for wave in waves if wave.negative]
        manager = ld.Manager()
        if mass_dependent:
            pos_amps = [wave.coefficient(manager) for wave in pos_r_waves]
        else:
            pos_amps = [wave.coefficient(manager) for wave in pos_r_waves]
        pos_amps = (
            [wave.coefficient(manager) for wave in pos_r_waves]
            if mass_dependent
            else [
                ld.amplitude_sum(wave.kmatrix(manager)) for wave in pos_r_waves
            ]
        )
        neg_amps = (
            [wave.coefficient(manager) for wave in neg_r_waves]
            if mass_dependent
            else [
                ld.amplitude_sum(wave.kmatrix(manager)) for wave in neg_r_waves
            ]
        )
        if phase_factor:
            k = Wave.phase_space_factor(manager) * manager.register(
                ld.Scalar(
                    'k_scale', ld.constant(1e8)
                )  # kappa tends to be ~1e-8
            )
            pos_amps = [k * a for a in pos_amps]
            neg_amps = [k * a for a in neg_amps]
        pos_zlms = [wave.zlm(manager) for wave in pos_r_waves]
        neg_zlms = [wave.zlm(manager) for wave in neg_r_waves]
        pos_re_terms = [c * zlm.real() for c, zlm in zip(pos_amps, pos_zlms)]
        pos_im_terms = [c * zlm.imag() for c, zlm in zip(pos_amps, pos_zlms)]
        neg_re_terms = [c * zlm.real() for c, zlm in zip(neg_amps, neg_zlms)]
        neg_im_terms = [c * zlm.imag() for c, zlm in zip(neg_amps, neg_zlms)]
        pos_re_sum = ld.amplitude_sum(pos_re_terms).norm_sqr()
        pos_im_sum = ld.amplitude_sum(pos_im_terms).norm_sqr()
        neg_re_sum = ld.amplitude_sum(neg_re_terms).norm_sqr()
        neg_im_sum = ld.amplitude_sum(neg_im_terms).norm_sqr()
        return manager.model(pos_re_sum + pos_im_sum + neg_re_sum + neg_im_sum)
