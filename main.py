from sklearn.datasets import make_moons
from src import models, plotter, utils
import tensorflow as tf
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=1500)
parser.add_argument("--steps", default=250)
parser.add_argument("--train_steps", default=100)
parser.add_argument("--resolution", default=50)
parser.add_argument("--schedule", default="continuous_geometric")
parser.add_argument("--model", default="VE")
parser.add_argument("--epochs", default=300)
parser.add_argument("--batch_size", default=1000)
parser.add_argument("--dataset", default="circle_gaussian")
args = parser.parse_args()
samples = args.samples
steps = args.steps
train_steps = args.train_steps
resolution = args.resolution
schedule_name = args.schedule
model_name = args.model
epochs = args.epochs
batch_size = args.batch_size
dataset = args.dataset

save_path = os.path.join(os.getcwd(), "figures")

gpu_device = tf.config.get_visible_devices("GPU")
if len(gpu_device) > 0:
    try:
        tf.config.set_logical_device_configuration(
            gpu_device[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    except Exception as e:
        raise e

schedule = utils.get_schedule(schedule_name)
train_config = {"steps": train_steps}
if model_name == "VE":
    model = models.SDEVE(schedule=schedule, train_config=train_config)
elif model_name == "VP":
    model = models.SDESubVP(schedule=schedule, train_config=train_config)
elif model_name == "subVP":
    model = models.SDESubVP(schedule=schedule, train_config=train_config)
else:
    raise NotImplementedError(
        f"Model {model_name} is not implemented,"
        f" choose from VE (Variance Exploding), VP (Variance Preserving), subVP (sub Variance Preserving) "
    )

if dataset == "circle_gaussian":
    x_train, y_train, distr = plotter.make_circle_gaussian(
        modes=4, sigma=0.5, radius=3, n_samples=1000
    )
elif dataset == "spiral":
    x_train, y_train = plotter.make_spiral_galaxy(4, 2, n_samples=1000, noise=0.2)
elif dataset == "cross":
    x_train, y_train, distr = plotter.make_cross_shaped_distribution(3000)
elif dataset == "moons":
    x_train, y_train = make_moons(3000, noise=0.1)
else:
    raise NotImplementedError(
        f"Dataset {dataset} not found. Choose from 'circle_gaussian', 'spiral', 'cross', 'moons'"
    )

save_path = os.path.join(
    save_path, model_name, dataset, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving figures to {save_path}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# model init
_ = model(x_train)
model.compile("adam")
model.fit(x_train, epochs=epochs, batch_size=batch_size)

xx, yy, xy = plotter.make_base_points((-6, 6), (-6, 6), resolution)
t = tf.linspace(0, 1, steps)
tt = tf.repeat(t[:, None], resolution**2, axis=1)
xxyy = tf.repeat(xy[None, :], steps, axis=0)
rtt = tf.reshape(tt, (-1, 1))
rxxyy = tf.reshape(xxyy, (-1, 2))
grad = model.grad_energy([rtt, rxxyy])
grad = grad.numpy().reshape(steps, resolution, resolution, 2)
energy = model.energy([rtt, rxxyy])
energy = energy.numpy().reshape(steps, resolution, resolution)

x_init = tf.random.normal(shape=(samples, 2)) * 2.5
bw, _, _ = model.backward_sde(x_init, steps=steps, deterministic=True)
bw = bw.numpy()
bws, _, _ = model.backward_sde(x_init, steps=steps, deterministic=False)
bws = bws.numpy()
fw, _, _ = model.forward_sde(x_train[:samples], steps=steps)
fw = fw.numpy()

_ = plotter.plot_trajectories3D(
    bws[:-10],
    save_path=save_path,
    name="backward_stochastic",
    title="Backward Trajectories",
)
_ = plotter.plot_trajectories3D(
    bw[:-10],
    save_path=save_path,
    name="backward_deterministic",
    title="Backward Trajectories",
)
_ = plotter.plot_trajectories3D(
    fw, save_path=save_path, name="forward_stochastic", title="Forward Trajectories"
)
_ = plotter.plot_gradient_field_and_energy(
    energy[:-10], grad[:-10], bw[10:], xx, yy, save_path, "grad&energy"
)
