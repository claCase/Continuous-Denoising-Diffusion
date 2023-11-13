import tensorflow as tf
from tensorflow_probability.python.distributions import (
    Normal,
    MultivariateNormalDiag,
    MultivariateNormalTriL,
)
from src.utils import noise, BetaSchedule
from typing import Union, Tuple


def sliced_score_estimator(grad, hess, noise_type="gaussian"):
    rnd = noise(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum(
        "bi,bio,bo->b", rnd, hess, rnd
    )  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.einsum("bi,bi->b", rnd, grad) ** 2  # eq. 8 trace estimator
    return sliced_hess + sliced_grad


def sliced_score_estimator_vr(grad, hess, noise_type="gaussian"):
    rnd = tf.random.normal(shape=tf.shape(hess)[:2], type=noise_type)
    sliced_hess = 0.5 * tf.einsum(
        "bi,bio,bo->b", rnd, hess, rnd
    )  # eq. 8 trace estimator
    sliced_grad = 0.5 * tf.linalg.norm(grad, axis=-1) ** 2
    return sliced_hess + sliced_grad


def score_loss(grad, hess, vr=False, noise_type="gaussian"):
    if vr:
        return tf.reduce_mean(sliced_score_estimator_vr(grad, hess, noise_type))
    return tf.reduce_mean(sliced_score_estimator(grad, hess, noise_type))


def noise_conditional_score_matching(
    grad: tf.Tensor, samples: tf.Tensor, corrupted_samples: tf.Tensor, sigma: float
):
    """
    Computes de-noising score estimation loss function
    :param grad: gradients of energy function of noised samples B x d
    :param samples: original uncorrupted samples os shape B x d
    :param corrupted_samples: noised samples of shape B x d
    :param sigma: standard deviation
    :return: B
    """
    targets = -(corrupted_samples - samples) / sigma**2
    loss = tf.reduce_sum(0.5 * (grad - targets) ** 2, -1) * sigma**2  # B
    loss = tf.reduce_mean(loss)
    return loss


def annealed_noise_conditional_score_matching(
    grad: tf.Tensor,
    samples: tf.Tensor,
    corrupted_samples: tf.Tensor,
    sigmas: Union[tf.Tensor, Tuple],
):
    """
    Computes de-noising score estimation loss function
    :param grad: gradients of energy function of noised samples B*|σ| x d
    :param samples: original uncorrupted samples os shape B x d
    :param corrupted_samples: noised samples of shape B*|σ| x d
    :param sigmas: standard deviation of shape |σ|
    :return: B
    """
    if not isinstance(sigmas, tf.Tensor):
        sigmas = tf.convert_to_tensor(sigmas, grad.dtype)
    ss = tf.shape(samples)
    B, d = ss[0], ss[1]
    s = tf.shape(sigmas)[0]
    r_corrupted = tf.reshape(corrupted_samples, (B, s, d))  # B x |σ| x d
    targets = -(r_corrupted - samples[:, None, :]) / sigmas[None, :, None] ** 2
    grad = tf.reshape(grad, (B, s, d))  # B x |σ| x d
    loss = (
        tf.reduce_sum(0.5 * (grad - targets) ** 2, -1) * sigmas[None, :] ** 2
    )  # B x |σ|
    loss = tf.reduce_mean(loss)
    return loss


def noise_conditional_score_matching_loss(grad, samples, corrupted_samples, sigmas):
    if isinstance(sigmas, float):
        tf.debugging.assert_shapes(
            [(grad, ("B", "d")), (samples, ("B", "d")), (corrupted_samples, ("B", "d"))]
        )
        return noise_conditional_score_matching(
            grad, samples, corrupted_samples, sigmas
        )
    elif isinstance(sigmas, tuple) or isinstance(sigmas, tf.Tensor):
        tf.debugging.assert_shapes(
            [
                (grad, ("BS", "d")),
                (corrupted_samples, ("BS", "d")),
                (samples, ("B", "d")),
            ]
        )
        return annealed_noise_conditional_score_matching(
            grad, samples, corrupted_samples, sigmas
        )


def prob0T_VE(data, noised_data, beta_schedule: BetaSchedule, t0, t1):
    """
    Computes Variance Exploding Kernel eq. 29  (assumes diagonal covariance matrix):
    p(t,x) = N(0, s(t) - s(0))
    :param data: tensor of shape (B,d)
    :param noised_data: tensor of shape (T, B,d)
    :param beta_schedule: BetaSchedule function used to noise data
    :param t0: initial time: 0
    :param t1: final time: 1
    :return: list[distribution_kernel, log_prob, grad_log_prob]
    """
    noised_data = tf.convert_to_tensor(noised_data)
    ds = tf.shape(noised_data)
    T, B, d = ds[0], ds[1], ds[2]
    dt = (t1 - t0) / T
    eval_t = tf.linspace(t0, t1, T)
    s0 = tf.cast(beta_schedule.f(t0), dtype=noised_data.dtype)
    s = tf.cast(beta_schedule.f(eval_t + dt), dtype=noised_data.dtype)
    kernel_mean = data  # B x d
    kernel_mean = tf.repeat(kernel_mean[None, :, :], T, axis=0)  # T x B x d
    kernel_variance = s - s0  # steps
    kernel_variance = tf.repeat(
        tf.repeat(kernel_variance[:, None], B, axis=1)[..., None], d, axis=-1
    )
    with tf.GradientTape() as tape:
        tape.watch(noised_data)
        norm_kernel = MultivariateNormalDiag(
            loc=kernel_mean, scale_diag=kernel_variance
        )
        lp = norm_kernel.log_prob(noised_data)
    grad_lp = tape.gradient(lp, noised_data)
    return norm_kernel, lp, grad_lp


def prob0T_VP(data, noised_data, beta_schedule: BetaSchedule, t0, t1):
    """
    Computes Variance Preserving Kernel eq. 29 (assumes diagonal covariance matrix):
    p(t,x) = N(0, s(t) - s(0))
    :param data: tensor of shape (B,d)
    :param noised_data: tensor of shape (T, B,d)
    :param beta_schedule: BetaSchedule function used to noise data
    :param t0: initial time: 0
    :param t1: final time: 1
    :param steps: n discretization steps
    :return: list[distribution_kernel, log_prob, grad_log_prob]
    """
    noised_data = tf.convert_to_tensor(noised_data)
    ds = tf.shape(noised_data)
    T, B, d = ds[0], ds[1], ds[2]
    eval_t = tf.linspace(t0, t1, T)
    s = tf.cast(beta_schedule.intf(eval_t), dtype=noised_data.dtype)
    s_mu = tf.math.exp(-1 / 2 * s)
    s_var = 1 - tf.math.exp(-s)
    kernel_mean = data * s_mu[:, None]  # B x d
    kernel_mean = tf.repeat(kernel_mean[None, :, :], T, axis=0)  # T x B x d
    kernel_variance = tf.repeat(
        tf.repeat(s_var[:, None][..., None], B, axis=1), d, axis=-1
    )  # T x B x d
    with tf.GradientTape() as tape:
        tape.watch(noised_data)
        norm_kernel = MultivariateNormalDiag(
            loc=kernel_mean, scale_diag=kernel_variance
        )
        lp = norm_kernel.log_prob(noised_data)
    grad_lp = tape.gradient(lp, noised_data)
    return norm_kernel, lp, grad_lp


def prob0T_sub_VP(data, noised_data, beta_schedule: BetaSchedule, t0, t1):
    """
    Computes Variance sub-Preserving Kernel eq. 29  (assumes diagonal covariance matrix):
    p(t,x) = N(0, s(t) - s(0))
    :param data: tensor of shape (B,d)
    :param noised_data: tensor of shape (T, B,d)
    :param beta_schedule: BetaSchedule function used to noise data
    :param t0: initial time: 0
    :param t1: final time: 1
    :param steps: n discretization steps
    :return:
    """
    noised_data = tf.convert_to_tensor(noised_data)
    ds = tf.shape(noised_data)
    T, B, d = ds[0], ds[1], ds[2]
    eval_t = tf.linspace(t0, t1, T)
    s = tf.cast(beta_schedule.intf(eval_t), dtype=noised_data.dtype)
    s_mu = tf.math.exp(-1 / 2 * s)
    s_var = (1 - tf.math.exp(-s)) ** 2
    kernel_mean = data * s_mu  # B x d
    kernel_mean = tf.repeat(kernel_mean, (T, 1, 1))
    kernel_variance = tf.repeat(s_var, (1, B, d))
    with tf.GradientTape() as tape:
        tape.watch(noised_data)
        norm_kernel = MultivariateNormalDiag(
            loc=kernel_mean, scale_diag=kernel_variance
        )
        lp = norm_kernel.log_prob(noised_data)
    grad_lp = tape.gradient(lp, noised_data)
    return norm_kernel, lp, grad_lp


def sde_noise_conditional_score_matching(score, true_grad, alpha=1.0):
    """
    Computes score matching with predetermined transition kernel
    :param score: T x B x d
    :param true_grad: T x B x d
    :param alpha: 1
    :return: 1
    """
    alpha = tf.convert_to_tensor(alpha, tf.float32)
    tf.debugging.assert_shapes([(score, ("T", "B", "D")), (true_grad, ("T", "B", "D"))])
    diff = tf.reduce_mean((score - true_grad) ** 2, (-1, -2))  # T
    # w = alpha / tf.reduce_mean(tf.linalg.norm(true_grad, axis=-1) ** 2, -1)  # T
    return tf.reduce_mean(diff)  # * w)
