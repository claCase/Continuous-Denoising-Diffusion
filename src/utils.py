from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from math import pi
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
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

from datetime import datetime


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
    samples = (
            np.random.normal(size=(modes, n_samples, 2)) * sigma + xy[:, None, :]
    )
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


def save_output_callback(
        model,
        inputs,
        save_path: os.path,
        every: int = 5,
        stop: int = 300,
        name: str = "default",
):
    now = datetime.now().isoformat()[:-7].replace(":", "_")
    save_path = os.path.join(save_path, now)
    os.makedirs(save_path)
    if isinstance(inputs, tf.Tensor):
        inputs = inputs.numpy()
    np.save(os.path.join(save_path, "inputs.npy"), inputs)

    def on_epoch_end(epoch, logs):
        if not epoch % every and epoch <= stop:
            o = model(inputs, training=True)
            if len(o) == 1:
                e, grad, hess = None, o, None
            elif len(o) == 3:
                e, grad, hess = o
            elif len(o) == 2:
                grad, z = o
                if z.shape[-1] == 1:
                    e = z
                    hess = None
                else:
                    hess = z
                    e = None
            else:
                raise NotImplementedError
            if e is not None:
                e = e.numpy()
                np.save(
                    os.path.join(save_path, str(epoch) + "_" + name + "_energy.npy"), e
                )
            if hess is not None:
                hess = hess.numpy()
                np.save(
                    os.path.join(save_path, str(epoch) + "_" + name + "_hessian.npy"),
                    hess,
                )
            np.save(
                os.path.join(save_path, str(epoch) + "_" + name + "_grad.npy"), grad
            )

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end), save_path


class AnnealNoiseSamples(tf.keras.callbacks.Callback):
    def __init__(self, start=2, anneal=0.1, type="linear"):
        super().__init__()
        self.anneal = tf.abs(anneal)
        self.type = type
        self.start = start
        self.lr = start

    def linear(self, i):
        self.lr = tf.maximum(self.start - self.anneal * i, 0.0)
        return self.lr

    def cosine(self, i):
        self.lr = tf.cos(i * pi / 4) ** 2 * self.sigmoid(i)
        return self.lr

    def sigmoid(self, i):
        self.lr = tf.maximum(2.0 * self.start / (1.0 + tf.exp(i) ** self.anneal), 0.0)

    def get_schedule(self, type):
        if type == "linear":
            return self.linear
        elif type == "cosine":
            return self.cosine
        elif type == "sigmoid":
            return self.sigmoid
        else:
            raise NotImplementedError(
                f"Type of schedule {type} is not supported, choose from 'linear', 'cosine', 'sigmoid'"
            )

    def on_epoch_end(self, epoch, logs=None):
        epoch = tf.cast(epoch, tf.float32)
        if hasattr(self.model, "anneal_samples"):
            self.model.anneal_samples = self.get_schedule(self.type)(epoch)
        else:
            raise AttributeError(f"Model does not have anneal_samples attribute")

    def on_train_end(self, logs=None):
        self.lr = self.start
        self.model.anneal_samples = 1.0


def noise(shape, type="gaussian"):
    rnd = tf.random.normal(shape=shape)
    if type == "gaussian":
        return rnd
    elif type == "rademacher":
        return tf.sign(rnd)
    elif type == "spherical":
        return (
                rnd
                / tf.linalg.norm(rnd, axis=-1, keepdims=True)
                * tf.math.sqrt(tf.shape(rnd)[-1])
        )
    else:
        raise NotImplementedError(
            f"Noise type must be in 'gaussian', 'rademacher', 'spherical'"
        )


def corrupt_samples(samples: tf.Tensor, sigmas: Union[Tuple, tf.Tensor]):
    """
    Corrupts samples based on σ parameter
    :param samples: samples to corrupt of shape B x d
    :param sigmas: standard deviation of normal distribution
    :return: corrupted samples of shape B*|σ| x d
    """
    ss = tf.shape(samples)
    B, d = ss[0], ss[1:]
    lens = tf.shape(sigmas)[0]
    sigmas = tf.convert_to_tensor(sigmas, samples.dtype)
    corruption = noise(shape=(B, lens, *tf.unstack(d)))
    sgs = tf.concat(((1, lens), tf.repeat(1, d.shape)), -1)
    corruption = corruption * tf.reshape(
        sigmas[None, :], sgs
    )  # scale noise by sigmas: B x |σ| x d
    corrupted_samples = samples[:, None, ...] + corruption
    corrupted_samples = tf.reshape(corrupted_samples, (-1, *tf.unstack(d)))  # B*|σ| x d
    return corrupted_samples


def log_boundary(x, x_lim=(0.0, 1.0)):
    constraint = tf.math.reduce_sum(tf.math.log(x - x_lim[0]), -1, keepdims=True)
    constraint += tf.math.reduce_sum(tf.math.log(x_lim[1] - x), -1, keepdims=True)
    return -constraint + 1e-9


def log_boundary2(x, b=(0, 0, -1, -1)):
    assert len(b) // x.shape[-1]
    x = tf.cast(x, tf.float32)
    b = tf.cast(b, tf.float32)
    b = tf.convert_to_tensor(b, tf.float32)
    A = tf.concat([tf.eye(len(b)), -tf.eye(len(b))])
    A = tf.cast(A, tf.float32)
    constraint = tf.math.log(A @ tf.transpose(x, (1, 0)) - b[:, None])
    return -tf.reduce_sum(tf.transpose(constraint, (1, 0)), -1, keepdims=True)


def dW(dt, shape):
    return tf.random.normal(stddev=tf.math.sqrt(dt), shape=shape)


def forward_dx(f, g, dt=1e-4, deterministic=False):
    """
    Returns the forward SDE function:
    dX = ƒ(X, t)*X + G(X,t)*dW

    :param f: drift function: (B, t, d) -> (B, d)
    :param g: diffusion function: (B, t, d) -> (B, 1) or (B, d) or (B, d, d)
    :param dt: delta time approximation
    :param deterministic: True if deterministic gradient flow, else computes forward SDE
    :return: SDE function of (t, x)
    """

    # actual time: t*dt
    def inner(t: int, x):
        """
        Computes the forward SDE
        :param t: current time step
        :param x: current state: (B, d)
        :return:  list[ x':(B, d), f(t, x):(B, d), g(t, X):(B, 1) or (B, d) or (B, d, d)]
        """
        d = tf.shape(x)[-1]
        ftx = f([tf.cast(t, x.dtype) * dt, x])
        if deterministic:
            return ftx
        else:
            gtx = g([tf.cast(t, x.dtype) * dt, x])
            gtx_shape = tf.shape(gtx)
            if gtx_shape.shape[0] == 3:
                return (
                    ftx * dt + tf.einsum("bio,bo->bi", gtx, dW(dt, x.shape)),
                    ftx,
                    gtx,
                )
            elif gtx_shape.shape[0] == 2:
                tf.assert_equal(
                    tf.logical_or(
                        tf.equal(gtx_shape[1], tf.constant(1, tf.int32)),
                        tf.equal(gtx_shape[1], d)),
                    True,
                    f"Shape of variance matrix {gtx_shape[1]} is not equal to input data dimension {d}"
                )
                updt = ftx * dt + gtx * dW(dt, x.shape)
                return updt, ftx, gtx
            else:
                raise RuntimeError(
                    f"Shape of covariance matrix is {gtx_shape} and is not supported,"
                    f" rank supported are 1, 2 and 3, covariance matrix rank is {gtx_shape.shape[0]}"
                )

    return inner


def backward_dx(f, g, grad_log_p, dt: float = 1e-4, deterministic: bool = False):
    """
    Computes the backward SDE:
    dX = {ƒ(X, t) - ∇[G(X,t)G(X,t)ʹ] - G(X,t)G(X,t)ʹ∇log p(x,t)} dt + G(X,t)dW

    :param f: drift function: (B, t, d) -> (B, d)
    :param g: diffusion function: (B, t, d) -> (B, 1) or (B, d) or (B, d, d)
    :param grad_log_p: log of p(t, x): (B, t, d) -> (B, d)
    :param dt: delta time approximation
    :param deterministic: True if deterministic gradient flow, else computes backward SDE
    :return: list[ x':(B, d), f(t, x):(B, d), g(t, X):(B, 1) or (B, d) or (B, d, d)]
    """

    # actual time: t*dt
    def inner(t, x):
        x = tf.convert_to_tensor(x, tf.float32)
        d = tf.shape(x)[-1]
        ftx = f([t * dt, x])
        with tf.GradientTape() as tape:
            tape.watch(x)
            gtx = g([t * dt, x])
            gtx_shape = tf.shape(gtx)
            if gtx_shape.shape[0] == 3:
                ss = tf.einsum("bio,bko->bik", gtx, gtx)
            elif gtx_shape.shape[0] == 2:
                tf.assert_equal(
                    tf.logical_or(
                        tf.equal(gtx_shape[1], tf.constant(1, tf.int32)),
                        tf.equal(gtx_shape[1], d)),
                    True,
                    f"Shape of variance matrix {gtx_shape[1]} is not equal to input data dimension {d}"
                )
                ss = gtx ** 2
            else:
                raise RuntimeError(
                    f"Shape of covariance matrix is {gtx_shape} and is not supported,"
                    f" rank supported are 1, 2 and 3, covariance matrix rank is {gtx_shape.shape[0]}"
                )
        grad_ss = tape.gradient(ss, x)
        if grad_ss is None:
            grad_ss = tf.convert_to_tensor(0.0, tf.float32)
        if deterministic:
            alpha = 1 / 2
        else:
            alpha = 1.0

        if gtx_shape.shape[0] == 3:
            rev_var = grad_ss + tf.einsum("bio,bo->bi", ss, grad_log_p([t * dt, x]))
        elif gtx_shape.shape[0] in {1, 2}:
            rev_var = grad_ss + ss * grad_log_p([tf.constant(t, tf.float32) * dt, x])
        else:
            raise RuntimeError

        dx = (ftx - alpha * rev_var) * dt
        if deterministic:
            return dx, ftx, gtx
        else:
            if gtx_shape.shape[0] == 3:
                go = tf.einsum("bio,bo->bi", gtx, dW(dt, x.shape))
            elif gtx_shape.shape[0] == 2:
                if gtx_shape[1] in [1, d]:
                    go = gtx * dW(dt, x.shape)
                else:
                    raise RuntimeError
            else:
                raise RuntimeError
            return dx + go, ftx, gtx

    return inner


def cond(max_steps=1000):
    def inner(t, *args, **kwargs):
        return tf.less(t, max_steps)

    return inner


def body(f):
    def b_(
            t, x, ta: tf.TensorArray, taf: tf.TensorArray = None, tag: tf.TensorArray = None
    ):
        fdx_, ftx, gtx = f(t, x)
        x1 = x + fdx_
        if taf is not None and tag is not None:
            return t + 1, x1, ta.write(t, x1), taf.write(t, ftx), tag.write(t, gtx)
        return t + 1, x1, ta.write(t, x1)

    return b_


def mixed_mode_colocation(n=0.01, v=0.0072168, a=-0.3872, b=-0.3251, c=1.17):
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


def sigmoid(c, k):
    def inner(t):
        return c / (1 + tf.math.exp(t * k))

    return inner


def d_sigmoid(c, k):
    def inner(t):
        return -(c * k * tf.math.exp(k * t)) / (tf.math.exp(k * t) + 1) ** 2

    return inner


def sigmoid_integral(c, k):
    def inner(t):
        return c * (t + tf.math.log(2.0) / k - tf.math.log(tf.math.exp(k * t) + 1) / k)

    return inner


def linear_interp(bmin, bmax, t0=0.0, t1=1.0):
    def inner(t):
        return bmin * (t - t1) / (t1 - t0) - bmax * (t0 - t) / (t1 - t0)

    return inner


def d_linear(bmin, bmax, t0=0.0, t1=1.0):
    def inner(t):
        return (bmin + bmax) / (t1 - t0) * tf.ones_like(t)

    return inner


def linear_integral(bmin, bmax, t0=0.0, t1=1.0):
    def inner(t):
        return (t * ((bmin + bmax) * t - 2 * bmin * t1 - 2 * bmax * t0)) / (
                2 * (t1 - t0)
        )

    return inner


class BetaSchedule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def f(self, t):
        raise NotImplementedError

    @abstractmethod
    def df(self, t):
        raise NotImplementedError

    @abstractmethod
    def intf(self, t):
        raise NotImplementedError


class SigmoidSchedule(BetaSchedule):
    def __init__(self, c=2.0, k=5.0):
        super().__init__()
        self.c = c
        self.k = k

    @tf.function
    def f(self, t):
        return self.c / 2 - sigmoid(self.c, self.k)(t) + 1e-3

    @tf.function
    def df(self, t):
        return -d_sigmoid(self.c, self.k)(t)

    @tf.function
    def intf(self, t):
        return self.c * t - sigmoid_integral(self.c, self.k)(t)


class LinearSchedule(BetaSchedule):
    def __init__(self, bmin=0.001, bmax=1.0, t0=0.0, t1=1.0):
        super().__init__()
        self.bmin = bmin
        self.bmax = bmax
        self.t0 = t0
        self.t1 = t1

    @tf.function
    def f(self, t):
        return linear_interp(self.bmin, self.bmax, self.t0, self.t1)(t)

    @tf.function
    def df(self, t):
        return d_linear(self.bmin, self.bmax, self.t0, self.t1)(t)

    @tf.function
    def intf(self, t):
        return linear_integral(self.bmin, self.bmax, self.t0, self.t1)(t)


class DiscreteGeometricSchedule(BetaSchedule):
    def __init__(self, bmin=0.1, bmax=10):
        super().__init__()
        self.bmin = bmin
        self.bmax = bmax

    @tf.function
    def f(self, t):
        return self.bmin + t * (self.bmax - self.bmin)

    @tf.function
    def intf(self, t):
        return 1 / 2 * t ** 2 * (self.bmax - self.bmin) - t * self.bmin

    @tf.function
    def df(self, t):
        return (self.bmax - self.bmin) * tf.ones_like(t)


class ContinuousGeometricSchedule(BetaSchedule):
    def __init__(self, bmin=0.1, bmax=10):
        super().__init__()
        self.bmin = bmin
        self.bmax = bmax

    @tf.function
    def f(self, t):
        return self.bmin * (self.bmax / self.bmin) ** t

    @tf.function
    def intf(self, t):
        return self.bmin ** 2 * (self.bmax / self.bmin) ** (2 * t)

    @tf.function
    def df(self, t):
        return (
                self.bmin
                * (self.bmax / self.bmin) ** t
                * tf.math.sqrt(2.0 * tf.math.log(self.bmax / self.bmin))
        )


def get_schedule(name):
    if isinstance(name, str):
        if name == "linear":
            return LinearSchedule()
        elif name == "sigmoid":
            return SigmoidSchedule()
        elif name == "discrete_geometric":
            return DiscreteGeometricSchedule()
        elif name == "continuous_geometric":
            return ContinuousGeometricSchedule()
        else:
            raise NotImplementedError(
                f"Schedule name {name} is not implemented, only ['linear', 'sigmoid', 'discrete_geometric',"
                f" 'continuous_geometric'] are supported"
            )
    elif isinstance(name, BetaSchedule):
        return name
    else:
        raise NotImplementedError(
            f"Type {type(name)} is not supported, only name in ['linear', 'sigmoid', 'discrete_geometric',"
            f" 'continuous_geometric'] or BetaSchedule instance"
        )


if __name__ == "__main__":
    import argparse

    plt.style.use("dark_background")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", default="mixed_mode")
    parser.add_argument("--t0", default=0)
    parser.add_argument("--t1", default=1)
    parser.add_argument("--n", default=600)
    parser.add_argument("--steps", default=int(1e4))
    args = parser.parse_args()
    ds = args.ds
    t = args.steps
    t0 = args.t0
    t1 = args.t1
    n = args.n
    dt = (t1 - t0) / t

    if ds == "mixed_mode":
        x_init = tf.random.uniform(shape=(n, 3), minval=-1, maxval=1)
        ta = tf.TensorArray(tf.float32, t, element_shape=x_init.shape)
        f, g = mixed_mode_colocation()
        fdx2 = forward_dx(f, g, dt=dt)
        bd2 = body(fdx2)
        cnd = cond(t)
        t_final, x_final, history = tf.while_loop(cnd, bd2, (0, x_init, ta))
        history = history.stack().numpy()
        marg_xy = history[:, :, [0, 1]].reshape(-1, 2)
        marg_xz = history[:, :, [0, 2]].reshape(-1, 2)
        marg_yz = history[:, :, [1, 2]].reshape(-1, 2)
        print("Plotting")
        fig, ax = plt.subplots(6)
        h, ex, ey = np.histogram2d(marg_xy[:, 0], marg_xy[:, 1], bins=[200, 200])
        h1, ex1, ey1 = np.histogram2d(marg_xz[:, 0], marg_xz[:, 1], bins=[200, 200])
        h2, ex2, ey2 = np.histogram2d(marg_yz[:, 0], marg_yz[:, 1], bins=[200, 200])
        ax[0].pcolormesh(ex, ey, h.T, norm="log", cmap="inferno")
        ax[1].pcolormesh(ex1, ey1, h1.T, norm="log", cmap="inferno")
        ax[2].pcolormesh(ex2, ey2, h2.T, norm="log", cmap="inferno")
        for i in range(n):
            ax[3].plot(history[:, i, 0], history[:, i, 1], color="blue", alpha=0.1)
            ax[4].plot(history[:, i, 0], history[:, i, 1], color="blue", alpha=0.1)
            ax[5].plot(history[:, i, 0], history[:, i, 1], color="blue", alpha=0.1)
        fig.tight_layout()
    else:
        d = 2
        I = tf.eye(d)
        cat = Categorical(probs=[0.5, 0.5])
        components = [
            MultivariateNormalDiag([-1, -1], [0.5, 0.5]),
            MultivariateNormalDiag([1, 1], [0.5, 0.5]),
        ]
        mix = Mixture(cat, components)
        x_init = mix.sample(n)
        ta = tf.TensorArray(tf.float32, t, element_shape=x_init.shape)
        fdx = forward_dx(
            lambda tx: np.zeros_like(tx[1]),
            lambda tx: tf.tile(I[None, :, :], (tx[1].shape[0], 1, 1)),
            dt=dt,
        )
        bd = body(fdx)
        cnd = cond(t)
        t_final, x_final, history = tf.while_loop(cnd, bd, (0, x_init, ta))
        history = history.stack().numpy()

        marg_x = np.concatenate(
            [np.tile(np.arange(t)[:, None], (1, n))[:, :, None], history[:, :, :1]], -1
        )
        marg_x = marg_x.reshape(-1, 2)
        h, ex, ey = np.histogram2d(marg_x[:, 0], marg_x[:, 1], bins=[1000, 100])
        print("Plotting")
        fig, ax = plt.subplots(2)
        ax[0].pcolormesh(ex, ey, h.T, norm="log", cmap="inferno")
        for i in range(n):
            ax[1].plot(np.arange(t), history[:, i, 0], color="blue", alpha=0.1)
        fig.tight_layout()
    plt.show()
