from src import models, plotter, utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

save_path = os.path.join(os.getcwd(), "figures")

x_train, y_train, distr = plotter.make_circle_gaussian(
    modes=4, sigma=0.5, radius=3, n_samples=1000
)
schedule = utils.ContinuousGeometricSchedule()
train_config = {"steps": 400}
model = models.SDEVE(schedule=schedule)
_ = model(x_train)
_ = model(x_train, forward=False)

model.compile("adam")
model.fit(x_train, epochs=300, batch_size=1000)

samples = 1500
steps = 200
points = 30
xx, yy, xy = plotter.make_base_points((-6, 6), (-6, 6), points)
t = tf.linspace(0, 1, steps)
tt = tf.repeat(t[:, None], points ** 2, axis=1)
xxyy = tf.repeat(xy[None, :], steps, axis=0)
rtt = tf.reshape(tt, (-1, 1))
rxxyy = tf.reshape(xxyy, (-1, 2))
grad = model.grad_energy([rtt, rxxyy])
grad = grad.numpy().reshape(steps, points, points, 2)
energy = model.energy([rtt, rxxyy])
energy = energy.numpy().reshape(steps, points, points)

x_init = tf.random.normal(shape=(samples, 2)) * 2.5
bw, _, _ = model.backward_sde(x_init, steps=steps, deterministic=True)
bws, _, _ = model.backward_sde(x_init, steps=steps, deterministic=False)
fw, _, _ = model.forward_sde(x_train[:samples], steps=steps)

_ = plotter.plot_trajectories(bws[:-10], save_path=save_path, name="backward_stochastic", title="Backward Trajectories")
_ = plotter.plot_trajectories(bw[:-10], save_path=save_path, name="backward_deterministic", title="Backward Trajectories")
_ = plotter.plot_trajectories(fw, save_path=save_path, name="forward_stochastic", title="Forward Trajectories")


"""
fig, ax = plt.subplots(1, figsize=(10, 10))
for i in range(steps):
    ax.clear()
    plt.scatter(bw[i, :, 0], bw[i, :, 1], color="white")
    ax.set_title(f"{i}")
    plt.pause(0.001)

fig, ax = plt.subplots(1, figsize=(10, 10))
for i in range(steps):
    ax.clear()
    plt.scatter(fw[i, :, 0], fw[i, :, 1], color="white")
    ax.set_title(f"{i}")
    plt.pause(0.001)


fig, ax = plt.subplots(2, 2, figsize=(20, 15))
ax[0,0].set_ylim(-10, 10)
ax[0,1].set_ylim(-10, 10)
ax[1,0].set_ylim(-10, 10)
ax[1,1].set_ylim(-10, 10)
for i in range(samples):
    ax[0, 0].plot(np.arange(steps), fw[:, i, 0], color="blue", alpha=0.1)
    ax[0, 1].plot(np.arange(steps), fw[:, i, 1], color="blue", alpha=0.1)
    ax[1, 0].plot(np.arange(steps), bws[:, i, 0], color="blue", alpha=0.1)
    ax[1, 1].plot(np.arange(steps), bws[:, i, 1], color="blue", alpha=0.1)
"""
