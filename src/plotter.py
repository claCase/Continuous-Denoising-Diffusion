from tensorflow_probability.python.distributions import Distribution
from scipy.ndimage import gaussian_filter
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
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
    Logistic,
    Distribution,
    MultivariateNormalDiag,
    NOT_REPARAMETERIZED,
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
            loc=[0, 2], scale_tril=tf.linalg.cholesky([[0.15 ** 2, 0], [0, 1]])
        ),
        MultivariateNormalTriL(
            loc=[-2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, 0.15 ** 2]])
        ),
        MultivariateNormalTriL(
            loc=[2, 0], scale_tril=tf.linalg.cholesky([[1, 0], [0, 0.15 ** 2]])
        ),
        MultivariateNormalTriL(
            loc=[0, -2], scale_tril=tf.linalg.cholesky([[0.15 ** 2, 0], [0, 1]])
        ),
    ]
    x = np.empty((n_samples * 4, 2))
    for i, c in enumerate(components):
        x[n_samples * i: n_samples * (i + 1), :] = c.sample(n_samples).numpy()
    y = np.repeat(np.arange(4), n_samples)
    mix = Mixture(
        cat=Categorical(probs=[1 / 4, 1 / 4, 1 / 4, 1 / 4]), components=components
    )
    return x, y, mix


def mixed_mode_colocation(n=0.01, v=0.0072168, a=-0.3872, b=-0.3251, c=1.17):
    """
    Defines drift and diffusion for mixed mode colocation process
    """

    def f(inputs):
        t, i = inputs
        x, y, z = i[:, :1], i[:, 1:2], i[:, -1:]
        dx = 1 / n * (y - x ** 2 - x ** 3)
        dy = z - x
        dz = -v - a * x - b * y - c * z
        return tf.concat([dx, dy, dz], -1)

    def g(inputs):
        t, i = inputs
        I = tf.eye(3)
        return tf.tile(I[None, :, :], (i.shape[0], 1, 1))

    return f, g


def make_grad_plot(
        model=None,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        num=50,
        reduce=5,
        ax=None,
        fig=None,
        iter=None,
        grad=None,
        e=None,
        xy=None,
        fontsize=20,
        quiver=True,
        contour=True,
        title=True,
        alpha=1,
        cmap="viridis",
):
    assert model is not None or (grad is not None and xy is not None)
    if model is not None:
        xx, yy, xy = make_base_points(x_lim, y_lim, num)
        o = model(xy, training=True)
        if len(o) == 3:
            grad, _, e = o
            e, grad = e.numpy(), grad.numpy()
        elif len(o) == 2:
            grad, z = o
            z = z.numpy()
            if z.shape[-1] == 1:
                e = z
            else:
                e = None
        else:
            raise ValueError(
                f"The model in training mode must return either (energy, grad, hessian) or (grad, hessian)"
            )
    else:
        if isinstance(grad, tf.Tensor):
            grad = grad.numpy()
        num = int(np.sqrt(xy.shape[0]))
        assert num ** 2 == xy.shape[0]
        xx, yy = np.split(xy, 2, -1)
        xx, yy = np.reshape(xx, (num, num)), np.reshape(yy, (num, num))
    if e is not None:
        title_ = "Estimated Vector Field parametrized by the Energy Model $ \\hat{\\bf E}_{\\theta} $ : $ \\nabla_{x} \\hat{ \\bf E}_{\\theta}(x) $"
        ee = e.reshape(num, num)
    else:
        title_ = "Estimated Vector Field of the Probability Distribution $ \\mathbb{\\hat{{P}}} $ : $ \\nabla_{x} \\mathbb{\\hat{{P}}}(x) $"

    if iter is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text = ax.text(
            xlim[0] - 0.05 * xlim[0],
            ylim[1] - 0.12 * ylim[1],
            f"Iteration {iter}",
            fontsize=15,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "black", "pad": 10},
        )

    assert grad.shape[-1] == 2
    dxx, dyy = np.split(grad, 2, -1)
    dxx, dyy = dxx.reshape(num, num), dyy.reshape(num, num)
    if ax is None:
        fig, ax = plt.subplots(1, dpi=300)
    if title:
        ax.set_title(title_)
    if e is not None and contour:
        if title:
            ax.set_title(title_, fontsize=fontsize)
        img = ax.contourf(xx, yy, ee, levels=100, cmap=cmap)
        ax.contour(
            xx, yy, ee, levels=20, colors="black", linestyles="solid", linewidths=0.51
        )
        if fig is not None:
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax1, orientation="vertical")
    if quiver:
        ax.quiver(
            xx[::reduce, ::reduce],
            yy[::reduce, ::reduce],
            dxx[::reduce, ::reduce],
            dyy[::reduce, ::reduce],
            alpha=alpha,
        )
    if fig is not None:
        return fig, ax
    return ax, e, xx, yy, dxx, dyy


def make_distribution_grad_plot(
        distr,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        num=200,
        reduce=10,
        ax=None,
        fig=None,
        fontsize=20,
):
    assert issubclass(type(distr), Distribution)
    xx, yy, xy = make_base_points(x_lim, y_lim, num)
    with tf.GradientTape() as tape:
        xy = tf.convert_to_tensor(xy, tf.float32)
        tape.watch(xy)
        ll = distr.log_prob(xy)
    grads = tape.gradient(ll, xy)
    dxx, dyy = tf.split(grads, 2, -1)
    dxx, dyy = tf.reshape(dxx, (num, num)).numpy(), tf.reshape(dyy, (num, num)).numpy()
    ll = ll.numpy().reshape(num, num)
    if ax is None:
        fig, ax = plt.subplots(1, dpi=300)
    ax.set_title(
        "True Vector Field of the probability distribution $\\mathbb{P} $: $ \\nabla_{x} \\mathbb{P}(x) $",
        fontsize=fontsize,
    )
    img = ax.contourf(xx, yy, ll, levels=100)
    ax.contour(
        xx, yy, ll, levels=20, colors="black", linestyles="solid", linewidths=0.51
    )
    ax.quiver(
        xx[::reduce, ::reduce],
        yy[::reduce, ::reduce],
        dxx[::reduce, ::reduce],
        dyy[::reduce, ::reduce],
    )
    if fig is not None:
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax1, orientation="vertical")
        return fig, ax
    return ax


def make_training_animation(
        save_path,
        dpi=150,
        fps=60,
        max_frames=None,
        fig=None,
        ax=None,
        name="default",
        **kwargs_grad_plot,
):
    save_name = name
    path, dirs, files = next(os.walk(save_path))
    epochs = set()
    types = set()
    names = set()
    for i in files:
        if i != "inputs.npy":
            try:
                splits = i.split("_")
                k = int(splits[0])
                obj = splits[-1].split(".")[0]
                epochs.update({k})
                types.update({obj})
                name = splits[1]
                names.update({name})
            except:
                pass
    epochs = list(epochs)
    epochs.sort()
    if max_frames is not None:
        max_frames = np.minimum(max_frames, len(epochs))
        epochs = epochs[:max_frames]
    try:
        inputs = np.load(os.path.join(save_path, "inputs.npy"))
    except Exception as e:
        raise e
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(15, 15), dpi=dpi)

    maxl = inputs.max(0)
    minl = inputs.min(0)
    x_lim = (minl[0], maxl[0])
    y_lim = (minl[1], maxl[1])
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    def plotter_grad(i, ax=ax):
        print(f"Processing frame {i}")
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        grad = np.load(
            os.path.join(save_path, str(epochs[i]) + "_" + name + "_grad.npy")
        )
        if "energy" in types:
            energy = np.load(
                os.path.join(save_path, str(epochs[i]) + "_" + name + "_energy.npy")
            )
        else:
            energy = None
        ax = make_grad_plot(
            grad=grad, e=energy, xy=inputs, ax=ax, iter=i, **kwargs_grad_plot
        )

    # fig.tight_layout()

    anim = animation.FuncAnimation(fig, plotter_grad, frames=len(epochs) - 1)
    anim.save(os.path.join(save_path, save_name + "_animation.gif"), fps=fps, dpi=dpi)


def plot_trajectories2D(
        ebm=None,
        trajectories=None,
        fig=None,
        ax=None,
        marg_x=None,
        marg_y=None,
        x_lim=(-10, 10),
        save_path=None,
        name="default_trajectory",
        dpi=90,
        distr=None,
        **kwargs_grad_plot,
):
    assert ebm is not None or trajectories is not None
    if trajectories is None:
        trajectories = ebm.langevin_dynamics(trajectories=True, n_samples=500)
    else:
        assert len(trajectories.shape) == 3

    if ax is None or fig is None:
        print(f"axis is {ax} and fig is {fig}")
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle("Langevin Dynamics Trajectory Sampling", fontsize=30)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        marg_x = fig.add_subplot(gs[0, 0], sharex=ax)
        marg_y = fig.add_subplot(gs[1, 1], sharey=ax)

    if marg_y is not None or marg_x is not None:
        title = False
        if distr is not None:
            assert issubclass(type(distr), Distribution)
            xx, yy, xy = make_base_points(x_lim, x_lim, 200)
            dp = distr.prob(xy).numpy()
            dp = dp.reshape(200, 200)
            px, py = dp.sum(0), dp.sum(1)
            px, py = px / px.sum(), py / py.sum()
            max_y_lim = np.maximum(np.max(px), np.max(py)) + 0.01
    else:
        title = True
    kwargs_grad_plot.update({"title": title})

    def plot_traj(i):
        print(f"Processing frame {i}")
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*x_lim)
        _ = make_grad_plot(ebm, x_lim, x_lim, ax=ax, iter=i, **kwargs_grad_plot)
        ax.scatter(trajectories[i, :, 0], trajectories[i, :, 1], s=5, color="black")
        if marg_y is not None:
            marg_y.clear()
            marg_y.set_xlim(0, max_y_lim)
            # max_ = np.minimum(trajectories.shape[1], 500)
            # marg_y.hist(trajectories[i, :max_, 0], bins=60, orientation='horizontal', density=True)
            h, b = np.histogram(trajectories[i, :, 1], bins=200, range=x_lim)
            h = h / h.sum()
            marg_y.stairs(h, b, fill=True, color="purple", orientation="horizontal")
            if distr is not None:
                marg_y.plot(py, xx[0], color="red")
        if marg_x is not None:
            marg_x.clear()
            marg_x.set_ylim(0, max_y_lim)
            # max_ = np.minimum(trajectories.shape[1], 500)
            h, b = np.histogram(trajectories[i, :, 0], bins=200, range=x_lim)
            h = h / h.sum()
            marg_x.stairs(h, b, fill=True, color="purple")
            # marg_x.hist(trajectories[i, :max_, 0], bins=60, density=True)
            if distr is not None:
                marg_x.plot(xx[0], px, color="red")

    anim = mpl.animation.FuncAnimation(fig, plot_traj, trajectories.shape[0])
    if save_path is not None:
        anim.save(os.path.join(save_path, name + "_animation.gif"), fps=60, dpi=dpi)
    return fig, ax, anim


def plot_trajectories(trajectories, t_max=100, particles=10, fig=None, save_path=None, name="deafult", title="",
                      axis_off=False):
    assert isinstance(trajectories, np.ndarray)
    steps = trajectories.shape[0]
    samples = trajectories.shape[1]
    p_id = np.random.choice(samples, particles, replace=False)
    h_tx, e_xx1, e_tt1 = np.histogram2d(
        trajectories[:, :, 0].flatten(), np.repeat(np.linspace(0, t_max, steps), samples),
        bins=(np.linspace(-12, 12, 200), np.linspace(0, t_max, steps + 1)),
        range=((-12, 12), (0, t_max))
    )
    tt1, xx1 = np.meshgrid(e_tt1[:-1], e_xx1[:-1])
    gtx = gaussian_filter(h_tx, sigma=1)
    # gtx = h_tx #gaussian_filter(h_tx, sigma=2)
    h_ty, e_yy, e_tt1 = np.histogram2d(
        trajectories[:, :, 1].flatten(), np.repeat(np.linspace(0, t_max, steps), samples),
        bins=(np.linspace(-12, 12, 200), np.linspace(0, t_max, steps + 1)),
        range=((-12, 12), (0, t_max))
    )
    gty = gaussian_filter(h_ty, sigma=1)
    # gty = h_ty
    h0, e_x, e_y = np.histogram2d(trajectories[0, :, 0], trajectories[0, :, 1], bins=(50, 50),
                                  range=((-12, 12.4), (-12, 12.4)))
    exx, eyy = np.meshgrid(e_x[:-1], e_y[:-1])
    h0 = gaussian_filter(h0, sigma=2)
    h1, e_x, e_y = np.histogram2d(trajectories[-1, :, 0], trajectories[-1, :, 1], bins=(50, 50),
                                  range=((-12, 12.4), (-12, 12.4)))
    h1 = gaussian_filter(h1, sigma=2)

    if fig is None:
        fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    t0, t1 = 0, steps - 1
    time = np.linspace(t0, t1, t1) * t_max / steps

    def draw_3D(i):
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
        ax.contourf(h1, exx, eyy, zdir="x", offset=t_max, cmap="inferno", zorder=0, alpha=0.5)
        for j in p_id:
            ax.plot(time[:i], trajectories[:i, j, 0], trajectories[:i, j, 1], color="lime", alpha=1, zorder=2,
                    linewidth=3)
            ax.plot(time[:i], np.zeros(i) + 12, trajectories[:i, j, 1], color="black", alpha=0.1, zorder=2)
            ax.plot(time[:i], trajectories[:i, j, 0], np.zeros(i) - 12, color="black", alpha=0.1, zorder=2)

        ax.scatter([time[i]] * particles, trajectories[i, p_id, 0], trajectories[i, p_id, 1], color="white", s=30,
                   zorder=2)
        ax.scatter([time[i]] * particles, trajectories[i, p_id, 0], np.zeros(particles) - 12, color="black", s=10,
                   zorder=2)
        ax.scatter([time[i]] * particles, np.zeros(particles) + 12, trajectories[i, p_id, 1], color="black", s=10,
                   zorder=2)

    anim = animation.FuncAnimation(fig, draw_3D, frames=(t1 - t0) - 1)
    if save_path is not None:
        anim.save(os.path.join(save_path, f"{name}_trajectory.gif"), fps=60, dpi=90)
    return fig, ax, anim


def plot_gradient_field_and_energy(energy, grad, points, xx, yy, save_path=None, name="default", fig=None):
    if fig is None:
        fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    steps = grad.shape[0]
    samples = grad.shape[1]

    def plot3D(i):
        ax.clear()
        ax.set_axis_off()
        ax.set_title(f"{i}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.contourf(xx, yy, energy[steps - i - 1], zdir="z", offset=-1, cmap="coolwarm")
        ax.plot_surface(xx, yy, energy[steps - i - 1],
                        color="royalblue", edgecolor="black", alpha=0.4, cstride=8, rstride=8)
        ax.quiver(
            xx,
            yy,
            np.zeros_like(xx) - 0.8,
            grad[steps - i - 1, :, :, 0],
            grad[steps - i - 1, :, :, 1],
            np.zeros_like(xx),
            length=.5,
            normalize=True,
            arrow_length_ratio=0.1,
            color="black",
            alpha=0.5,
            linewidths=2
        )
        ax.scatter(points[i, :, 0], points[i, :, 1], np.zeros(samples) - 0.8, s=10, color="white")

    anim = animation.FuncAnimation(fig, plot3D, frames=steps - 1)
    if save_path is not None:
        anim.save(os.path.join(save_path, f"{name}_grad&energy.gif"), fps=60, dpi=100)
    return fig, ax, anim


def plot_grad(grad, points, xx, yy, name="default", save_path=None):
    fig, ax = plt.subplots(1, figsize=(15, 15))
    steps = grad.shape[0]

    def sanim(i):
        ax.clear()
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.quiver(xx, yy, grad[steps - i - 1, :, :, 0], grad[steps - i - 1, :, :, 1], color="blue")
        ax.scatter(points[i, :, 0], points[i, :, 1], color="white", s=3)

    anim = animation.FuncAnimation(fig, sanim, frames=steps - 1)
    if save_path is not None:
        anim.save(os.path.join(save_path, f"{name}_grad_animation.gif"), fps=60, dpi=100)
    plt.close()
