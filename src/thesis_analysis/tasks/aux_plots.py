import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np
from scipy.integrate import quad
from scipy.special import sph_harm

from thesis_analysis.paths import Paths


class MakeAuxiliaryPlots(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return [
            luigi.LocalTarget(Paths.plots / 'argand_diagram.png'),
            luigi.LocalTarget(Paths.plots / 'spherical_harmonics.png'),
        ]

    def run(self):
        argand_path = self.output()[0].path
        spherical_harmonics_path = self.output()[1].path

        mpl_style.use('thesis_analysis.thesis')

        # Argand diagram
        m_a = np.array([1.2, 1.4])
        gamma_a = np.array([0.2, 0.1])

        fig, ax = plt.subplots(ncols=2)

        m = np.linspace(0.900, 2.0, 1000)
        t_k = [eval_t_k(s, m_a, gamma_a) for s in np.power(m, 2)]
        t_bw = [eval_t_bw(s, m_a, gamma_a) for s in np.power(m, 2)]

        ax[0].plot(
            m, np.power(np.abs(t_k), 2), color='k', ls='-', label='K-Matrix'
        )
        ax[0].plot(
            m,
            np.power(np.abs(t_bw), 2),
            color='k',
            ls=':',
            label='Breit-Wigner',
        )
        ax[0].vlines(
            m_a,
            0,
            1,
            transform=ax[0].get_xaxis_transform(),
            colors='r',
            lw=1,
            label='Resonance Masses',
        )
        ax[0].set_xlabel(r'Mass (GeV/$c^2$)')
        ax[0].set_ylabel(r'$|T|^2$')
        ax[0].set_xlim(0.9, 2.0)
        ax[0].set_ylim(0.0)
        ax[0].legend()

        cx, cy = 0, 0.5
        r = 0.5
        padx, pady = 0.2, 0.4

        circle = plt.Circle((cx, cy), r, color='w')

        w, h = 2 * (r + padx), 2 * (r + pady)
        x0, y0 = cx - r - padx, cy - r - pady
        rect = plt.Rectangle(
            (x0, y0),
            w,
            h,
            facecolor='none',
            hatch='//',
            edgecolor='k',
            alpha=0.5,
        )
        ax[1].add_patch(rect)
        ax[1].add_patch(circle)

        ax[1].plot(
            np.real(t_k), np.imag(t_k), color='k', ls='-', label='K-Matrix'
        )
        ax[1].plot(
            np.real(t_bw),
            np.imag(t_bw),
            color='k',
            ls=':',
            label='Breit-Wigner',
        )
        ax[1].set_xlim(x0, x0 + w)
        ax[1].set_ylim(y0, y0 + h)
        ax[1].set_xlabel(r'$\Re[T]$')
        ax[1].set_ylabel(r'$\Im[T]$')
        ax[1].legend()
        fig.savefig(argand_path)
        plt.close()

        # Spherical harmonics plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lms = [(0, 0), (2, 0), (2, 1), (2, 2)]
        ylabels = ['$S_0$', '$D_0$', '$D_1$', '$D_2$']
        xs = np.linspace(-1, 1, 100)

        for i, lm in enumerate(lms):
            ys = np.ones_like(xs) * i
            zs = np.array([int_ylm_abs_square(*lm, x) for x in xs])
            ax.plot(xs, ys, zs)
            ax.fill_between(xs, ys, zs, xs, ys, np.zeros_like(zs), alpha=0.2)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(ylabels)
        ax.set_xlabel(r'$\cos\theta$')
        ax.set_zlabel(r'$|Y_\ell^m|^2$')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        plt.savefig(spherical_harmonics_path)


def eval_t_k(
    s: complex, m_a: np.ndarray, gamma_a: np.ndarray
) -> np.complexfloating:
    g_a = np.sqrt(gamma_a / m_a)
    k = np.sum(np.power(g_a, 2) / (np.power(m_a, 2) - s))
    return k / (1 - 1j * k)


def eval_t_bw(
    s: complex, m_a: np.ndarray, gamma_a: np.ndarray
) -> np.complexfloating:
    bw_a = (m_a * gamma_a) / (np.power(m_a, 2) - s - 1j * m_a * gamma_a)
    return np.sum(bw_a)


def ylm(ell: int, m: int, costheta: float, phi: float) -> complex:
    return sph_harm(m, ell, phi, np.arccos(costheta))


def ylm_abs_square(ell: int, m: int, costheta: float, phi: float) -> float:
    return float(np.power(np.abs(ylm(ell, m, costheta, phi)), 2))


def int_ylm_abs_square(ell: int, m: int, costheta: float) -> float:
    def f(x: float) -> float:
        return ylm_abs_square(ell, m, costheta, x)

    return quad(f, 0, 2 * np.pi)[0]
