from src import models, losses, plotter, utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, distr = plotter.make_circle_gaussian(modes=4, sigma=0.5, radius=3, n_samples=1000)
schedule = utils.SigmoidSchedule(10, 5)
model = models.SDE_VE(schedule=schedule)
_ = model(x_train)
_ = model(x_train, forward=False)

model.compile("adam")
model.fit(x_train, epochs=100, batch_size=1000)

steps = 100
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

samples = 500
bw, _, _ = model.backward_sde(tf.random.normal(shape=(samples, 2)), steps=steps, deterministic=True)
fw, _, _ = model.forward_sde(x_train[:samples], steps=steps)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for i in range(steps):
    ax.clear()
    ax.set_title(f"{i}")
    ax.plot_surface(xx, yy, energy[i])
    ax.quiver(xx, yy, np.zeros_like(xx) - 0.8, grad[i, :, :, 0], grad[i, :, :, 1], np.zeros_like(xx), length=3.1,
              arrow_length_ratio=0.01, color="blue")
    ax.scatter(x_train[:500, 0], x_train[:500, 1], np.zeros(500) - 0.8)
    plt.pause(0.001)


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


fig, ax = plt.subplots(2, 2)
for i in range(samples):
    ax[0, 0].plot(np.arange(steps), fw[:, i, 0], color="blue", alpha=0.1)
    ax[0, 1].plot(np.arange(steps), fw[:, i, 1], color="blue", alpha=0.1)
    ax[1, 0].plot(np.arange(steps), bw[:, i, 0], color="blue", alpha=0.1)
    ax[1, 1].plot(np.arange(steps), bw[:, i, 1], color="blue", alpha=0.1)
