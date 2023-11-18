from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import numpy as np
from numpy.random import choice
from sklearn.utils import shuffle as skshuffle
from tensorflow_probability.python.distributions import (
    MultivariateNormalTriL,
    Mixture,
    Categorical,
    MultivariateNormalDiag,
)

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
plt.style.use("dark_background")


def make_base_points(x_lim=(-5, 5), y_lim=(-5, 5), num=200):
    x = np.linspace(*x_lim, num=num)
    y = np.linspace(*y_lim, num=num)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1)
    return xx, yy, xy


def make_spiral_galaxy(
    n_spirals=5, length=1, angle=np.pi / 2, n_samples=100, noise=0, shuffle=True
):
    thetas = np.linspace(0, np.pi * 2, n_spirals + 1)
    thetas = thetas[:-1]
    radius = np.linspace(
        np.zeros(len(thetas)) + 0.1, np.ones(len(thetas)) * length + 0.1, n_samples
    )
    angles = np.linspace(thetas, thetas + angle, n_samples)
    if noise:
        angles += (
            np.random.normal(size=angles.shape)
            * noise
            * np.linspace(1.5, 0.1, n_samples)[:, None]
        )
    x0 = np.cos(angles) * radius
    x1 = np.sin(angles) * radius
    x0 = x0.T.reshape(-1, 1)
    x1 = x1.T.reshape(-1, 1)
    xy = np.concatenate([x0, x1], -1)
    y = np.repeat(np.arange(n_spirals), n_samples)
    if shuffle:
        xy, y = skshuffle(xy, y)
    return xy, y


def make_circle_gaussian(modes=5, sigma=1, radius=2, n_samples=100, shuffle=True):
    thetas = np.linspace(0, np.pi * 2, modes + 1)
    thetas = thetas[:-1]
    x0 = np.cos(thetas) * radius
    x1 = np.sin(thetas) * radius
    x0 = x0.T.reshape(-1, 1)
    x1 = x1.T.reshape(-1, 1)
    xy = np.concatenate([x0, x1], -1).astype(np.float32)
    components = [MultivariateNormalDiag(mu, [sigma, sigma]) for mu in xy]
    probs = tf.ones(modes) / modes
    cat = Categorical(probs=probs)
    mix = Mixture(cat=cat, components=components)
    samples = np.random.normal(size=(modes, n_samples, 2)) * sigma + xy[:, None, :]
    samples = samples.reshape(-1, 2)
    y = np.repeat(np.arange(modes), n_samples)
    if shuffle:
        xy, y = skshuffle(samples, y)
    return xy, y, mix


def make_cross_shaped_distribution(n_samples):
    components = [
        MultivariateNormalTriL(
            loc=[0, 2], scale_tril=tf.linalg.cholesky([[0.15**2, 0], [0, 1]])
        ),
        MultivariateNormalTriL(
            loc=[-2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, 0.15**2]])
        ),
        MultivariateNormalTriL(
            loc=[2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, 0.15**2]])
        ),
        MultivariateNormalTriL(
            loc=[0, -2], scale_tril=tf.linalg.cholesky([[0.15**2, 0], [0, 1]])
        ),
    ]
    x = np.empty((n_samples * 4, 2))
    for i, c in enumerate(components):
        x[n_samples * i : n_samples * (i + 1), :] = c.sample(n_samples).numpy()
    y = np.repeat(np.arange(4), n_samples)
    mix = Mixture(
        cat=Categorical(probs=[1 / 4, 1 / 4, 1 / 4, 1 / 4]), components=components
    )
    return x, y, mix


def mixed_mode_colocation(n=0.01, v=0.0072168, a=-0.3872, b=-0.3251, c=1.17):
    """
    Defines drift and diffusion for mixed mode co-location process
    """

    def f(inputs):
        t, i = inputs
        x, y, z = i[:, :1], i[:, 1:2], i[:, -1:]
        dx = 1 / n * (y - x**2 - x**3)
        dy = z - x
        dz = -v - a * x - b * y - c * z
        return tf.concat([dx, dy, dz], -1)

    def g(inputs):
        t, i = inputs
        I = tf.eye(3)
        return tf.tile(I[None, :, :], (i.shape[0], 1, 1))

    return f, g


def plot_trajectories3D(
    trajectories,
    t_max=100,
    particles=10,
    fig=None,
    save_path=None,
    name="deafult",
    title="",
    axis_off=False,
):
    assert isinstance(trajectories, np.ndarray)
    steps = trajectories.shape[0]
    samples = trajectories.shape[1]
    p_id = np.random.choice(samples, particles, replace=False)
    h_tx, e_xx1, e_tt1 = np.histogram2d(
        trajectories[:, :, 0].flatten(),
        np.repeat(np.linspace(0, t_max, steps), samples),
        bins=(np.linspace(-12, 12, 200), np.linspace(0, t_max, steps + 1)),
        range=((-12, 12), (0, t_max)),
    )
    tt1, xx1 = np.meshgrid(e_tt1[:-1], e_xx1[:-1])
    gtx = gaussian_filter(h_tx, sigma=1)
    h_ty, e_yy, e_tt1 = np.histogram2d(
        trajectories[:, :, 1].flatten(),
        np.repeat(np.linspace(0, t_max, steps), samples),
        bins=(np.linspace(-12, 12, 200), np.linspace(0, t_max, steps + 1)),
        range=((-12, 12), (0, t_max)),
    )
    gty = gaussian_filter(h_ty, sigma=1)
    h0, e_x, e_y = np.histogram2d(
        trajectories[0, :, 0],
        trajectories[0, :, 1],
        bins=(50, 50),
        range=((-12, 12.4), (-12, 12.4)),
    )
    exx, eyy = np.meshgrid(e_x[:-1], e_y[:-1])
    h0 = gaussian_filter(h0, sigma=2)
    h1, e_x, e_y = np.histogram2d(
        trajectories[-1, :, 0],
        trajectories[-1, :, 1],
        bins=(50, 50),
        range=((-12, 12.4), (-12, 12.4)),
    )
    h1 = gaussian_filter(h1, sigma=2)

    if fig is None:
        fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    t0, t1 = 0, steps - 1
    time = np.linspace(t0, t1, t1) * t_max / steps

    def draw_3D(i):
        print(f"Frame {i}")
        ax.clear()
        ax.set_xlim(0, t_max)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        if title != "":
            ax.text(5, 5, 20, title, fontsize=35)
        if axis_off:
            ax.set_axis_off()
        else:
            ax.xaxis.set_pane_color(color=(0, 0, 0, 0), alpha=0)
            ax.yaxis.set_pane_color(color=(0, 0, 0, 0), alpha=0)
            ax.zaxis.set_pane_color(color=(0, 0, 0, 0), alpha=0)
            ax.xaxis.line.set_color(color=(0, 0, 0, 0))
            ax.yaxis.line.set_color(color=(0, 0, 0, 0))
            ax.zaxis.line.set_color(color=(0, 0, 0, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_ylabel("X", color="white", fontsize=20)
            ax.set_zlabel("Y", color="white", fontsize=20)
            ax.set_xlabel("Time", color="white", fontsize=20, labelpad=20)
        ax.contourf(tt1, xx1, gtx, zdir="z", offset=-12, cmap="inferno", zorder=0)
        ax.contourf(tt1, gty, xx1, zdir="y", offset=12, cmap="inferno", zorder=0)
        ax.contourf(h0, exx, eyy, zdir="x", offset=0, cmap="inferno", zorder=0)
        ax.contourf(
            h1, exx, eyy, zdir="x", offset=t_max, cmap="inferno", zorder=0, alpha=0.5
        )
        for j in p_id:
            ax.plot(
                time[:i],
                trajectories[:i, j, 0],
                trajectories[:i, j, 1],
                color="lime",
                alpha=1,
                zorder=2,
                linewidth=3,
            )
            ax.plot(
                time[:i],
                np.zeros(i) + 12,
                trajectories[:i, j, 1],
                color="black",
                alpha=0.1,
                zorder=2,
            )
            ax.plot(
                time[:i],
                trajectories[:i, j, 0],
                np.zeros(i) - 12,
                color="black",
                alpha=0.1,
                zorder=2,
            )

        ax.scatter(
            [time[i]] * particles,
            trajectories[i, p_id, 0],
            trajectories[i, p_id, 1],
            color="white",
            s=30,
            zorder=2,
        )
        ax.scatter(
            [time[i]] * particles,
            trajectories[i, p_id, 0],
            np.zeros(particles) - 12,
            color="black",
            s=10,
            zorder=2,
        )
        ax.scatter(
            [time[i]] * particles,
            np.zeros(particles) + 12,
            trajectories[i, p_id, 1],
            color="black",
            s=10,
            zorder=2,
        )

    anim = animation.FuncAnimation(fig, draw_3D, frames=(t1 - t0) - 1)
    if save_path is not None:
        anim.save(os.path.join(save_path, f"{name}_trajectory.gif"), fps=60, dpi=90)
    return fig, ax, anim


def plot_gradient_field_and_energy(
    energy, grad, points, xx, yy, save_path=None, name="default", fig=None
):
    assert energy.shape[0] == grad.shape[0] == points.shape[0]
    if fig is None:
        fig = plt.figure(figsize=(10, 15))
    ax = plt.subplot(211, projection="3d")
    ax.computed_zorder = False
    ax2 = plt.subplot(212)
    ax2.set_box_aspect(1)
    steps = grad.shape[0]
    samples = points.shape[1]

    def plot3D(i):
        print(f"Frame {i}")
        ax.clear()
        ax.set_axis_off()
        # ax.set_title(f"{i}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.contourf(
            xx, yy, energy[steps - i - 1], zdir="z", offset=-1, cmap="winter", zorder=0
        )
        ax.plot_surface(
            xx,
            yy,
            energy[steps - i - 1],
            color="cyan",
            edgecolor="black",
            alpha=0.5,
            cstride=5,
            rstride=5,
            zorder=3,
        )
        ax.quiver(
            xx,
            yy,
            np.zeros_like(xx) - 0.8,
            grad[steps - i - 1, :, :, 0],
            grad[steps - i - 1, :, :, 1],
            np.zeros_like(xx),
            length=0.5,
            normalize=True,
            arrow_length_ratio=0.1,
            color="black",
            alpha=0.5,
            linewidths=2,
            zorder=1,
        )
        ax.scatter(
            points[i, :, 0],
            points[i, :, 1],
            np.zeros(samples),
            s=3,
            color="white",
            zorder=3,
        )
        ax2.clear()
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.quiver(
            xx,
            yy,
            grad[steps - i - 1, :, :, 0],
            grad[steps - i - 1, :, :, 1],
            color="dodgerblue",
        )
        ax2.scatter(points[i, :, 0], points[i, :, 1], color="white", s=5)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, plot3D, frames=steps - 1)
    if save_path is not None:
        anim.save(os.path.join(save_path, f"{name}_grad&energy.gif"), fps=60, dpi=100)
    return fig, ax, anim


def plot_particle_stream(trajectories, t_max=100):
    steps = trajectories.shape[0]
    samples = trajectories.shape[1]
    h_tx, e_xx, e_tt = np.histogram2d(
        trajectories[:, :, 0].flatten(),
        np.repeat(np.linspace(0, t_max, steps), samples),
        bins=(np.linspace(-12, 12, 200), np.linspace(0, t_max, steps + 1)),
        range=((-12, 12), (0, t_max)),
    )
    gtx = gaussian_filter(h_tx, sigma=1)
    fig, ax = plt.subplots(2, figsize=(15, 10))

    ax[0].contourf(e_tt, e_xx, gtx.T, cmap="inferno")
    for i in range(samples):
        ax.clear()
        ax[1].scatter(
            np.linspace(0, 1, steps), trajectories[:, i, 1], color="blue", alpha=0.1
        )
    return fig, ax


def stack_gifs(path1, path2):
    from PIL import Image

    fw = []
    fw_gif = Image.open(path1)
    bw_gif = Image.open(path2)
    bw = []
    for i in range(bw_gif.n_frames):
        fw_gif.seek(i)
        bw_gif.seek(i)
        fw_i = fw_gif.convert("RGBA")
        bw_i = bw_gif.convert("RGBA")
        fw.append(fw_i)
        bw.append(bw_i)

    fig, ax = plt.subplots(2, 1, figsize=(10, 21))

    def stack(i):
        print(f"frame {i}")
        ax[0].clear()
        ax[1].clear()
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].imshow(fw[i])
        ax[1].imshow(bw[i])
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, stack, bw_gif.n_frames)
    anim.save("./forward_backward_stacked.gif", fps=60, dpi=80)
