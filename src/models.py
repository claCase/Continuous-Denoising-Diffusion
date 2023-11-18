from abc import abstractmethod
import numpy as np
import tensorflow as tf
from typing import Tuple, Union
from src.losses import (
    sde_noise_conditional_score_matching,
    sub_VP_kernel,
    VP_kernel,
    VE_kernel,
    log_prob_and_grad,
)
from src.utils import (
    corrupt_samples,
    forward_dx,
    body,
    backward_dx,
    cond,
    BetaSchedule,
    get_schedule,
    dW,
)

Layer, SCNN2D, Drop, BN, Concatenate = (
    tf.keras.layers.Layer,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.Dropout,
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.Concatenate,
)
Dense, AvgPool2D, GAvgPool2D, Lambda = (
    tf.keras.layers.Dense,
    tf.keras.layers.AvgPool2D,
    tf.keras.layers.GlobalAvgPool2D,
    tf.keras.layers.Lambda,
)
Model, Sequential = tf.keras.models.Model, tf.keras.models.Sequential


class EBMDenoisingFlow(Model):
    def __init__(
        self, train_config: dict = None, schedule: Union[str, BetaSchedule] = "linear"
    ):
        super().__init__()
        base_train_cnfg = dict(t0=0.0, t1=1.0, steps=1000, deterministic=False)
        self.train_cnfg = train_config
        if self.train_cnfg is None:
            self.train_cnfg = base_train_cnfg
        else:
            base_train_cnfg.update(train_config)
            self.train_cnfg = base_train_cnfg

        assert all(
            [
                k in {"t0", "t1", "steps", "deterministic"}
                for k in self.train_cnfg.keys()
            ]
        )

        self.schedule = get_schedule(schedule)  # noise variance schedule

    @abstractmethod
    def make_energy_model(self, input_shape):
        raise NotImplementedError

    @abstractmethod
    def make_mean_f(self, input_shape):
        raise NotImplementedError

    @abstractmethod
    def make_variance_g(self, input_shape):
        raise NotImplementedError

    def build(self, input_shape):
        self.e = self.make_energy_model(input_shape)
        self.e.build(input_shape)
        self.f = self.make_mean_f(input_shape)
        self.f.build(input_shape)
        self.g = self.make_variance_g(input_shape)
        self.g.build(input_shape)
        self.built = True

    def grad_energy(self, inputs):
        it, ix = inputs
        with tf.GradientTape() as tape:
            tape.watch(ix)
            energy = self.e(inputs)
        grad_e_x = tape.gradient(energy, ix)
        return grad_e_x

    def energy(self, inputs):
        return self.e(inputs)

    def call(self, inputs, training=False, forward=True, mask=None, **kwargs):
        ix = inputs
        if forward:
            return self.forward_sde(ix, **self.train_cnfg, **kwargs)
        else:
            return self.backward_sde(ix, **self.train_cnfg, **kwargs)

    # @tf.function
    def forward_sde(
        self,
        x_init,
        steps: int = 1000,
        t0: float = 0.0,
        t1: float = 1.0,
        deterministic=False,
    ):
        x_shape = x_init.shape
        B = x_shape[0]
        d = x_shape[1]
        dt = (t1 - t0) / steps
        ta = tf.TensorArray(tf.float32, steps, element_shape=x_init.shape)
        taf = tf.TensorArray(tf.float32, steps, element_shape=x_init.shape)
        tag = tf.TensorArray(tf.float32, steps, element_shape=(B, d))
        fdx = forward_dx(self.f, self.g, dt, deterministic)
        bd = body(fdx)
        stop = cond(max_t=t1 - dt, min_t=t0 - dt)
        k_final, t_final, x_final, history, history_f, history_g = tf.while_loop(
            stop, bd, (t0, 0, x_init, ta, taf, tag)
        )
        return history.stack(), history_f.stack(), history_g.stack()

    # @tf.function
    def backward_sde(
        self,
        x_init=None,
        steps: int = 1000,
        t0: float = 0.0,
        t1: float = 1.0,
        deterministic=False,
    ):
        if x_init is None:
            x_init = tf.random.normal(shape=(500, self.f.output_shape[-1]))
        x_shape = x_init.shape
        B = x_shape[0]
        d = x_shape[1]
        dt = (t1 - t0) / steps
        ta = tf.TensorArray(tf.float32, steps, element_shape=x_init.shape)
        taf = tf.TensorArray(tf.float32, steps, element_shape=x_init.shape)
        tag = tf.TensorArray(tf.float32, steps, element_shape=(B, d))
        bdx = backward_dx(self.f, self.g, self.grad_energy, dt, deterministic)
        bd = body(bdx)
        stop = cond(min_t=t0 + dt, max_t=t1 + dt)
        k_final, t_final, x_final, history, history_f, history_g = tf.while_loop(
            stop, bd, (t1, 0, x_init, ta, taf, tag)
        )
        return history.stack(), history_f.stack(), history_g.stack()

    @abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    def predictor(self, t, x):
        score = self.grad_energy([t, x])
        s = self.g([t, x])
        x1 = x - (s * score + dW(s, x.shape))
        return x1

    def corrector(self, t, x, r=1.0):
        score = self.grad_energy([t, x])
        z = tf.random.normal(shape=x.shape)
        eps = (
            2
            * (
                r
                * tf.reduce_mean(
                    tf.linalg.norm(z, axis=-1, keepdims=True)
                    / tf.linalg.norm(score, axis=-1, keepdims=True),
                    keepdims=True,
                )
            )
            ** 2
        )
        x1 = x - (eps * score + tf.math.sqrt(2 * eps) * z)
        return x1

    def predictor_corrector(
        self, x=None, N=None, M=20, r=1e-3, particles=500, return_seq=True
    ):
        t0, t1 = self.train_cnfg["t0"], self.train_cnfg["t1"]
        if N is None:
            N = self.train_cnfg["steps"]
        if x is None:
            d = self.e.input_shape[-1][-1]
            var = self.g([(self.train_cnfg["t1"],), tf.constant((1,))])
            x = tf.random.normal(shape=(particles, d)) * var

        T = tf.linspace(t0, t1, N)
        if return_seq:
            ta = tf.TensorArray(tf.float32, N, element_shape=x.shape)
        for n in range(N, 0, -1):
            t = T[n - 1]
            for m in range(M):
                x = self.corrector((t,), x, r)
            x = self.predictor((t,), x)
            if return_seq:
                ta = ta.write(n - 1, x)
        if return_seq:
            return ta.stack()
        return x


class SDEVE(EBMDenoisingFlow):
    def __init__(self, hidden_layers=(100, 100, 10), activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.activation = activation

    def make_mean_f(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        zeros_lambda = Lambda(lambda i: tf.zeros_like(i[-1]))
        o = zeros_lambda([i1, i2])
        return Model([i1, i2], o)

    def make_variance_g(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        lambda_var = Lambda(
            lambda i: tf.math.sqrt(self.schedule.df(i[0]))
            * tf.ones_like(i[1], dtype=i[1].dtype)
        )
        o = lambda_var([i1, i2])
        return Model([i1, i2], o)

    def make_energy_model(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(*x[1:],))
        output_dim = 1
        layers = []
        for h in self.hidden_layers:
            layers.append(Dense(h, self.activation))
        layers.append(Dense(output_dim, activation="elu"))
        conc = Concatenate(axis=-1)
        x = conc([i1 * tf.ones_like(i2), i2])
        for l in layers:
            x = l(x)
        return Model([i1, i2], x)

    def train_step(self, data):
        t0, t1, steps = (
            self.train_cnfg.get("t0"),
            self.train_cnfg.get("t1"),
            self.train_cnfg.get("steps"),
        )
        ds = tf.shape(data)
        T, B, d = steps, ds[0], ds[1]
        # generate noised data from perturbation kernel:
        # you can do this because the sde has closed form solution for transition kernel, and the score must learn the
        # grad of the closed form transition kernel, this way it's much faster to train the network since you don't
        # need to evaluate the forward sde
        noise_kernel, eval_t = VE_kernel(data, t0, t1, steps, self.schedule)
        noised_data = tf.squeeze(noise_kernel.sample(1), 0)  # T x B x d
        resh_noised_data = tf.reshape(noised_data, (-1, d))  # T*B x d
        t = tf.repeat(
            eval_t[:, None, None], B, axis=1
        )  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}


class SDEVP(EBMDenoisingFlow):
    def __init__(self, hidden_layers=(100, 100, 10), activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.activation = activation

    def make_mean_f(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        mean_lambda = Lambda(lambda i: -1 / 2 * self.schedule.df(i[0]) * i[1])
        o = mean_lambda([i1, i2])
        return Model([i1, i2], o)

    def make_variance_g(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        lambda_var = Lambda(
            lambda i: tf.math.sqrt(self.schedule.df(i[0]))
            * tf.ones_like(i[1], dtype=i[1].dtype)
        )
        o = lambda_var([i1, i2])
        return Model([i1, i2], o)

    def make_energy_model(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(*x[1:],))
        output_dim = 1
        layers = []
        for h in self.hidden_layers:
            layers.append(Dense(h, self.activation))
        layers.append(Dense(output_dim, activation="elu"))
        conc = Concatenate(axis=-1)
        x = conc([i1 * tf.ones_like(i2), i2])
        for l in layers:
            x = l(x)
        return Model([i1, i2], x)

    def train_step(self, data):
        t0, t1, steps = (
            self.train_cnfg.get("t0"),
            self.train_cnfg.get("t1"),
            self.train_cnfg.get("steps"),
        )
        ds = tf.shape(data)
        T, B, d = steps, ds[0], ds[1]
        # generate noised data from perturbation kernel:
        # you can do this because the sde has closed form solution for transition kernel, and the score must learn the
        # grad of the closed form transition kernel, this way it's much faster to train the network since you don't
        # need to evaluate the forward sde
        noise_kernel, eval_t = VP_kernel(data, t0, t1, steps, self.schedule)
        noised_data = tf.squeeze(noise_kernel.sample(1), 0)  # T x B x d
        resh_noised_data = tf.reshape(noised_data, (-1, d))  # T*B x d
        t = tf.repeat(
            eval_t[:, None, None], B, axis=1
        )  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}


class SDESubVP(EBMDenoisingFlow):
    def __init__(self, hidden_layers=(100, 100, 10), activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.activation = activation

    def make_mean_f(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        mean_lambda = Lambda(lambda i: -1 / 2 * self.schedule.df(i[0]) * i[1])
        o = mean_lambda([i1, i2])
        return Model([i1, i2], o)

    def make_variance_g(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(batch_input_shape=(1,)), tf.keras.Input((*x[1:],))
        lambda_var = Lambda(
            lambda i: tf.math.sqrt(
                self.schedule.f(i[0]) * (1 - tf.math.exp(-2 * self.schedule.intf(i[0])))
            )
            * tf.ones_like(i[1], dtype=i[1].dtype)
        )
        o = lambda_var([i1, i2])
        return Model([i1, i2], o)

    def make_energy_model(self, input_shape):
        x = input_shape
        i1, i2 = tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(*x[1:],))
        output_dim = 1
        layers = []
        for h in self.hidden_layers:
            layers.append(Dense(h, self.activation))
        layers.append(Dense(output_dim, activation="elu"))
        conc = Concatenate(axis=-1)
        x = conc([i1 * tf.ones_like(i2), i2])
        for l in layers:
            x = l(x)
        return Model([i1, i2], x)

    def train_step(self, data):
        t0, t1, steps = (
            self.train_cnfg.get("t0"),
            self.train_cnfg.get("t1"),
            self.train_cnfg.get("steps"),
        )
        ds = tf.shape(data)
        T, B, d = steps, ds[0], ds[1]
        # generate noised data from perturbation kernel:
        # you can do this because the sde has closed form solution for transition kernel, and the score must learn the
        # grad of the closed form transition kernel, this way it's much faster to train the network since you don't
        # need to evaluate the forward sde
        noise_kernel, eval_t = sub_VP_kernel(data, t0, t1, steps, self.schedule)
        noised_data = tf.squeeze(noise_kernel.sample(1), 0)  # T x B x d
        resh_noised_data = tf.reshape(noised_data, (-1, d))  # T*B x d
        t = tf.repeat(
            eval_t[:, None, None], B, axis=1
        )  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}
