from abc import abstractmethod
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union
from src.losses import (
    score_loss,
    noise_conditional_score_matching_loss,
    sde_noise_conditional_score_matching,
    sub_VP_kernel, VP_kernel, VE_kernel, log_prob_and_grad
)
from src.utils import (
    corrupt_samples,
    forward_dx,
    body,
    backward_dx,
    cond,
    dW,
    BetaSchedule,
    get_schedule,
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


class SlicedScoreMatching(Model):
    """
    https://proceedings.mlr.press/v115/song20a.html
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation
    """

    def __init__(
            self,
            hidden_layers: Tuple[int, ...] = (100, 50),
            output_dim: int = 2,
            activation: str = "relu",
            vr=False,
            noise_type="gaussian",
            anneal=0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_dim = output_dim
        self.vr = vr
        self.noise_type = noise_type
        self.anneal = anneal
        self.f = None

    def build(self, input_shape):
        self.f = self.make_score_model(
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            output_dim=input_shape[-1],
        )
        self.f.build(input_shape)
        self.built = True

    def call(self, inputs, training=False, mask=None):
        if training:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                grad = self.f(inputs)
            hess = tape.batch_jacobian(grad, inputs)
            return grad, hess
        return self.f(inputs)

    def train_step(self, data):
        data += tf.random.normal(shape=tf.shape(data)) * self.anneal
        with tf.GradientTape() as tape:
            grad, hess = self(data, training=True)
            loss = score_loss(grad, hess, vr=self.vr, noise_type=self.noise_type)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    def langevin_dynamics(
            self,
            initial_points=None,
            steps=500,
            x_lim=(-6, 6),
            n_samples=100,
            trajectories=False,
    ):
        try:
            in_dim = self.f.layers[0].input_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        def alpha(i):
            return 1 / (1000 + i)

        if initial_points is None:
            x = tf.random.uniform(
                minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, in_dim)
            )
        else:
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps, n_samples, in_dim))
            traj[0, :, :] = x

        prog = tf.keras.utils.Progbar(steps - 1)
        for t in range(1, steps):
            a = alpha(t)
            x = (
                    x
                    + 0.5 * a * self(x)
                    + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], in_dim))
            )
            if trajectories:
                traj[t, :, :] = x.numpy()
            prog.update(t)
        if trajectories:
            return traj
        return x

    def annealed_langevin_dynamics(
            self,
            initial_points=None,
            steps=500,
            n_samples=100,
            x_lim=(-6.0, 6.0),
            sigma_high=1.0,
            sigma_low=0.01,
            levels=10,
            e=0.0001,
            trajectories=False,
    ):
        try:
            in_dim = self.f.layers[0].input_shape[1:]
        except AttributeError as ae:
            raise ae
        except RuntimeError as re:
            raise re

        alphas = tf.exp(
            tf.linspace(tf.math.log(sigma_low), tf.math.log(sigma_high), levels)
        )[::-1]

        if initial_points is None:
            x = tf.random.uniform(
                minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, *in_dim)
            )
        else:
            assert initial_points.shape == in_dim
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps * levels + 1, n_samples, *in_dim))
            traj[0, :, :] = x
        cntr = 1
        prog = tf.keras.utils.Progbar(levels * steps)
        for l in range(levels):
            a = e * alphas[l] / alphas[-1]
            for t in range(0, steps):
                x = (
                        x
                        + 0.5 * a * self(x)
                        + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], *in_dim))
                )
                if trajectories:
                    traj[cntr, :, :] = x.numpy()
                prog.update(cntr)
                cntr += 1
        if trajectories:
            return traj
        return x

    @staticmethod
    def make_score_model(hidden_layers, activation, output_dim):
        i = tf.keras.layers.Input(shape=(output_dim,))
        for k, h in enumerate(hidden_layers):
            l = Dense(h, activation)
            if k == 0:
                x = l(i)
            else:
                x = l(x)
        l = Dense(output_dim, activation="linear")
        o = l(x)
        return Model(i, o)


class EBMSlicedScoreMatching(SlicedScoreMatching):
    """
    https://proceedings.mlr.press/v115/song20a.html
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation
    """

    def __init__(
            self,
            hidden_layers: Tuple[int, ...] = (100, 50),
            activation: str = "relu",
            vr=False,
            noise_type="gaussian",
            anneal=0.0,
            **kwargs
    ):
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            anneal=anneal,
            output_dim=1,
            noise_type=noise_type,
            vr=vr,
            **kwargs
        )

    def build(self, input_shape):
        # super(Model, self).build(input_shape)
        self.f = self.make_score_model(
            hidden_layers=self.hidden_layers, activation=self.activation
        )
        self.f.build(input_shape)
        self.built = True

    def call(self, inputs, training=False, mask=None):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                e = self.f(inputs)
            grad = tape1.gradient(e, inputs)
        hess = tape.batch_jacobian(grad, inputs)
        if training:
            return grad, hess, e
        return grad

    def train_step(self, data):
        if self.anneal:
            data += tf.random.normal(shape=tf.shape(data)) * self.anneal
        with tf.GradientTape() as tape:
            e, grad, hess = self(data, training=True)
            loss = score_loss(grad, hess, vr=self.vr, noise_type=self.noise_type)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    @staticmethod
    def make_score_model(hidden_layers, activation, output_dim=1):
        layers = []
        for h in hidden_layers:
            layers.append(Dense(h, activation))
        layers.append(Dense(output_dim, activation="elu"))
        return Sequential(layers)


class NoiseConditionalScoreModel(Model):
    """
    https://arxiv.org/abs/1907.05600
    Generative Modeling by Estimating Gradients of the Data Distribution
    """

    def __init__(
            self,
            sigma=0.5,
            hidden_layers: Tuple[int, ...] = (100, 50),
            output_dim: int = 2,
            activation: str = "relu",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.loss_tracker = tf.keras.metrics.Mean(name="score")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_dim = output_dim

    @staticmethod
    def make_score_model(hidden_layers, activation, input_dim):
        layers = []
        for h in hidden_layers:
            layers.append(Dense(h, activation))
        layers.append(Dense(input_dim, activation="linear"))
        return Sequential(layers)

    def build(self, input_shape):
        # super().build(input_shape)
        self.f = self.make_score_model(
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            input_dim=input_shape[-1],
        )
        self.f.build(input_shape)
        self.built = True

    def call(self, inputs, training=False, mask=None):
        return self.f(inputs, training=training)

    def train_step(self, data):
        B = tf.shape(data)[0]
        sg_s = (self.sigma,) if isinstance(self.sigma, float) else self.sigma
        lens = len(sg_s)
        corrupted_samples = corrupt_samples(data, sg_s)
        with tf.GradientTape() as tape:
            o = self(corrupted_samples, training=True)
            if len(o) == 2:
                grad, e = o
            else:
                grad = o

            data = tf.reshape(data, (B, -1))
            grad = tf.reshape(grad, (B * lens, -1))
            corrupted_samples = tf.reshape(corrupted_samples, (B * lens, -1))

            loss = noise_conditional_score_matching_loss(
                grad, data, corrupted_samples, self.sigma
            )
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"Score": self.loss_tracker.result()}

    def langevin_dynamics(
            self,
            initial_points=None,
            steps=500,
            x_lim=(-6, 6),
            n_samples=100,
            trajectories=False,
    ):
        try:
            in_dim = self.f.layers[0].input_shape[-1]
        except AttributeError as e:
            raise e
        except RuntimeError as e:
            raise e

        def alpha(i):
            return 1 / (1000 + i)

        if initial_points is None:
            x = tf.random.uniform(
                minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, in_dim)
            )
        else:
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps, n_samples, in_dim))
            traj[0, :, :] = x

        prog = tf.keras.utils.Progbar(steps - 1)
        for t in range(1, steps):
            a = alpha(t)
            x = (
                    x
                    + 0.5 * a * self(x)
                    + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], in_dim))
            )
            if trajectories:
                traj[t, :, :] = x.numpy()
            prog.update(t)
        if trajectories:
            return traj
        return x

    def annealed_langevin_dynamics(
            self,
            initial_points=None,
            steps=500,
            n_samples=100,
            x_lim=(-6.0, 6.0),
            sigma_high=1.0,
            sigma_low=0.01,
            levels=10,
            e=0.0001,
            trajectories=False,
    ):
        try:
            in_dim = self.f.layers[0].input_shape[1:]
        except AttributeError as ae:
            raise ae
        except RuntimeError as re:
            raise re

        alphas = tf.exp(
            tf.linspace(tf.math.log(sigma_low), tf.math.log(sigma_high), levels)
        )[::-1]

        if initial_points is None:
            x = tf.random.uniform(
                minval=x_lim[0], maxval=x_lim[1], shape=(n_samples, *in_dim)
            )
        else:
            assert initial_points.shape == in_dim
            x = initial_points

        if trajectories:
            traj = np.empty(shape=(steps * levels + 1, n_samples, *in_dim))
            traj[0, :, :] = x
        cntr = 1
        prog = tf.keras.utils.Progbar(levels * steps)
        for l in range(levels):
            a = e * alphas[l] / alphas[-1]
            for t in range(0, steps):
                x = (
                        x
                        + 0.5 * a * self(x)
                        + tf.math.sqrt(a) * tf.random.normal(shape=(x.shape[0], *in_dim))
                )
                if trajectories:
                    traj[cntr, :, :] = x.numpy()
                prog.update(cntr)
                cntr += 1
        if trajectories:
            return traj
        return x


class EBMNoiseConditionalScoreModel(NoiseConditionalScoreModel):
    """
    https://arxiv.org/abs/1907.05600
    Generative Modeling by Estimating Gradients of the Data Distribution

    forward SDE: dX = ƒ(X, t)*X + G(X,t)*dW
    backward SDE: {ƒ(X, t) - ∇[G(X,t)G(X,t)ʹ] - G(X,t)G(X,t)ʹ∇log p(x,t)} dt + G(X,t)dW

    sΘ(x, t) = ∇log p(x,t) = ∇log EΘ(x,t), the score is learned and parametrized by the gradient of the energy
    """

    def __init__(
            self,
            sigmas: Union[Tuple, tf.Tensor] = tf.linspace(0.001, 1, 10)[::-1],
            hidden_layers: Tuple[int, ...] = (100, 50),
            activation: str = "relu",
            **kwargs
    ):
        super().__init__(
            hidden_layers=hidden_layers, activation=activation, sigma=sigmas, **kwargs
        )

        self.f = self.make_score_model(
            hidden_layers=self.hidden_layers, activation=self.activation
        )

    @staticmethod
    def make_score_model(hidden_layers, activation, output_dim=1):
        layers = []
        for h in hidden_layers:
            layers.append(Dense(h, activation))
        layers.append(Dense(output_dim, activation="elu"))
        return Sequential(layers)

    def build(self, input_shape):
        super(Model, self).build(input_shape)
        self.f.build(input_shape)

    def call(self, inputs, training=False, mask=None):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            e = self.f(inputs)
        grad = tape.gradient(e, inputs)
        if training:
            return grad, e
        else:
            return grad


class EBMDenoisingFlow(Model):
    def __init__(
            self, train_cnfg: dict = None, schedule: Union[str, BetaSchedule] = "linear"
    ):
        super().__init__()
        base_train_cnfg = dict(t0=0.0, t1=1.0, steps=1000, deterministic=False)
        self.train_cnfg = train_cnfg
        if self.train_cnfg is None:
            self.train_cnfg = base_train_cnfg
        else:
            base_train_cnfg.update(train_cnfg)
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


class SDE_VE(EBMDenoisingFlow):
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
            lambda i: tf.math.sqrt(self.schedule.df(i[0])) *
                      tf.ones_like(i[1], dtype=i[1].dtype)
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
        t = tf.repeat(eval_t[:, None, None], B, axis=1)  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}


class SDE_VP(EBMDenoisingFlow):
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
        t = tf.repeat(eval_t[:, None, None], B, axis=1)  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}


class SDE_SubVP(EBMDenoisingFlow):
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
        t = tf.repeat(eval_t[:, None, None], B, axis=1)  # each batch must have same time
        t = tf.reshape(t, (-1, 1))  # T*B x d
        with tf.GradientTape() as tape:
            score = self.grad_energy([t, resh_noised_data])  # TxB x d
            score = tf.reshape(score, (T, B, d))
            lp, grad_lp = log_prob_and_grad(noise_kernel, noised_data)  # T x B x d
            loss = sde_noise_conditional_score_matching(score, grad_lp)
        grads = tape.gradient(loss, self.e.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.e.trainable_variables))
        return {"Loss": loss}
