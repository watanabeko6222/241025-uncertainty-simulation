import numpy as np

from dataclasses import dataclass

import matplotlib.pyplot as plt
plt.style.use('fast')
plt.rcParams["figure.figsize"] = [6.4,4.0]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.autolayout"] = False
plt.rcParams["figure.subplot.left"] = 0.14  
plt.rcParams["figure.subplot.bottom"] = 0.14
plt.rcParams["figure.subplot.right"] =0.90  
plt.rcParams["figure.subplot.top"] = 0.91   
plt.rcParams["figure.subplot.wspace"] = 0.20
plt.rcParams["figure.subplot.hspace"] = 0.20

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["grid.color"] = "black"

plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fancybox"] = False

@dataclass
class GaussianVariable:
    mean: float
    std: float

@dataclass
class SampleState:
    x: float
    y: float
    theta: float
    vel: float

class GaussianState:
    def __init__(self, x: GaussianVariable, y: GaussianVariable, theta: GaussianVariable, vel: GaussianVariable):
        self.x = x
        self.y = y
        self.theta = theta
        self.vel = vel

        xs = np.random.normal(x.mean, x.std, 10000)
        ys = np.random.normal(y.mean, y.std, 10000)
        thetas = np.random.normal(theta.mean, theta.std, 10000)
        vels = np.random.normal(vel.mean, vel.std, 10000)
        
        self.samples = [SampleState(x, y, theta, vel) for x, y, theta, vel in zip(xs, ys, thetas, vels)]

# egoveichle_o = GaussianState(
#     x=GaussianVariable(mean=5, std=3.00),
#     y=GaussianVariable(mean=5, std=3.01),
#     theta=GaussianVariable(mean=0, std=0.28),
#     vel=GaussianVariable(mean=(25 / 3.6), std=0.19),
# )
# egoveichle_o = GaussianState(
#     x=GaussianVariable(mean=5, std=0.72),
#     y=GaussianVariable(mean=5, std=1.39),
#     theta=GaussianVariable(mean=0, std=0.01),
#     vel=GaussianVariable(mean=(25 / 3.6), std=0.03),
# )
# egoveichle_o = GaussianState(
#      x=GaussianVariable(mean=5, std=2.45),
#      y=GaussianVariable(mean=5, std=2.83),
#      theta=GaussianVariable(mean=0, std=0.26),
#      vel=GaussianVariable(mean=(25 / 3.6), std=0.08),
#  )
egoveichle_o = GaussianState(
    x=GaussianVariable(mean=5, std=1.63),
    y=GaussianVariable(mean=5, std=2.86),
    theta=GaussianVariable(mean=0, std=0.11),
    vel=GaussianVariable(mean=(25 / 3.6), std=0.04),
)

# cyclyst_o = GaussianState(
#     x=GaussianVariable(mean=25, std=2.0),
#     y=GaussianVariable(mean=0, std=2.0),
#     theta=GaussianVariable(mean=np.pi, std=0.1),
#     vel=GaussianVariable(mean=(15 / 3.6), std=0.1),
# )

cyclyst_e = GaussianState(
    x=GaussianVariable(mean=20, std=0.20),
    y=GaussianVariable(mean=-5, std=0.20),
    theta=GaussianVariable(mean=0, std=0.2),
    vel=GaussianVariable(mean=(40/3.6), std=0.125),
)

def transition(state: SampleState, dt: float) -> SampleState:
    return SampleState(
        x=state.x + state.vel * np.cos(state.theta) * dt,
        y=state.y + state.vel * np.sin(state.theta) * dt,
        theta=state.theta,
        vel=state.vel
    )

def jacobian_transition(state: SampleState, dt: float) -> np.ndarray:
    jacobian = np.array([
        [1, 0, -state.vel * np.sin(state.theta) * dt, np.cos(state.theta) * dt],
        [0, 1, state.vel * np.cos(state.theta) * dt, np.sin(state.theta) * dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return jacobian

def transform_e2o(o_e: SampleState, e_c: SampleState) -> SampleState:
    return SampleState(
        x=o_e.x + e_c.x * np.cos(o_e.theta) - e_c.y * np.sin(o_e.theta),
        y=o_e.y + e_c.x * np.sin(o_e.theta) + e_c.y * np.cos(o_e.theta),
        theta=o_e.theta + e_c.theta,
        vel=np.sqrt(o_e.vel**2 + e_c.vel**2 - 2 * o_e.vel * e_c.vel * np.cos(e_c.theta))
    )

def jacoboan_transform_e2o(o_e: SampleState, e_c: SampleState) -> np.ndarray:
    jacobian_o_e = np.array([
        [1, 0, -e_c.x * np.sin(o_e.theta) - e_c.y * np.cos(o_e.theta), 0],
        [0, 1, e_c.x * np.cos(o_e.theta) - e_c.y * np.sin(o_e.theta), 0],
        [0, 0, 1, 0],
        [0, 0, 0, (1/2) * (1/np.sqrt(o_e.vel**2 + e_c.vel**2 - 2 * o_e.vel * e_c.vel * np.cos(e_c.theta))) * (2 * o_e.vel - 2 * e_c.vel * np.cos(e_c.theta))]
    ])
    
    jacobian_e_c = np.array([
        [np.cos(o_e.theta), -np.sin(o_e.theta), 0, 0],
        [np.sin(o_e.theta), np.cos(o_e.theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, (1/2) * (1/np.sqrt(o_e.vel**2 + e_c.vel**2 - 2 * o_e.vel * e_c.vel * np.cos(e_c.theta))) * (2 * e_c.vel + 2 * o_e.vel * np.sin(e_c.theta))]
    ])
    
    return jacobian_o_e, jacobian_e_c
    
o_c_list: list[SampleState] = []
o_c_future_list: list[SampleState] = []

for o_e, e_c in zip(egoveichle_o.samples, cyclyst_e.samples):
    o_c = transform_e2o(o_e, e_c)
    o_c_future = transition(o_c, 5.0)
    o_c_list.append(o_c)
    o_c_future_list.append(o_c_future)

o_e_mean = SampleState(
    x=egoveichle_o.x.mean,
    y=egoveichle_o.y.mean,
    theta=egoveichle_o.theta.mean,
    vel=egoveichle_o.vel.mean
)
e_c_mean = SampleState(
    x=cyclyst_e.x.mean,
    y=cyclyst_e.y.mean,
    theta=cyclyst_e.theta.mean,
    vel=cyclyst_e.vel.mean
)
o_c_mean = transform_e2o(o_e_mean, e_c_mean)
o_c_future_mean = transition(o_c_mean, 5.0)

jacobian_o_e, jacobian_e_c = jacoboan_transform_e2o(o_e_mean, e_c_mean)
o_e_Sigma = np.array([
    [egoveichle_o.x.std**2, 0, 0, 0],
    [0, egoveichle_o.y.std**2, 0, 0],
    [0, 0, egoveichle_o.theta.std**2, 0],
    [0, 0, 0, egoveichle_o.vel.std**2]
])
e_c_Sigma = np.array([
    [cyclyst_e.x.std**2, 0, 0, 0],
    [0, cyclyst_e.y.std**2, 0, 0],
    [0, 0, cyclyst_e.theta.std**2, 0],
    [0, 0, 0, cyclyst_e.vel.std**2]
])

o_c_Sigma = jacobian_o_e @ o_e_Sigma @ jacobian_o_e.T + jacobian_e_c @ e_c_Sigma @ jacobian_e_c.T
o_c_future_Sigma = jacobian_transition(o_c_mean, 5.0) @ o_c_Sigma @ jacobian_transition(o_c_mean, 5.0).T
print(o_e_Sigma)
print(e_c_Sigma)
print(o_c_Sigma)

# plot

plt.figure(figsize=(10, 6))
# ratio of x and y is 1:1
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-10, 40)
plt.ylim(-20, 20)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.scatter([o_c.x for o_c in o_c_list], [o_c.y for o_c in o_c_list], c='tab:orange', marker='x', alpha=0.1)
plt.scatter([o_e.x for o_e in egoveichle_o.samples], [o_e.y for o_e in egoveichle_o.samples], c='tab:blue', marker='x',alpha=0.1)
# plt.scatter([o_c_future.x for o_c_future in o_c_future_list], [o_c_future.y for o_c_future in o_c_future_list], c='g', marker='x', label='o_c_future',alpha=0.1)
plt.scatter(egoveichle_o.x.mean, egoveichle_o.y.mean, c='black', marker='*', label='ego-vehicle mean')
plt.scatter(o_c_mean.x, o_c_mean.y, c='black', marker='h', label='cyclist mean')
# plt.scatter(o_c_future_mean.x, o_c_future_mean.y, c='black', marker='*', label='o_c_future_mean')
# 
#eliipse
for Sigma, mean in zip([o_c_Sigma], [o_c_mean]):
    P = Sigma[:2, :2]
    w, v = np.linalg.eig(P)
    angle = np.arctan2(v[1, 0], v[0, 0])
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=[mean.x, mean.y],
        width=2 * np.sqrt(w[0]) * 2.4477,
        height=2 * np.sqrt(w[1]) * 2.4477,
        angle=np.rad2deg(angle),
        edgecolor='black',
        facecolor='none',
        linestyle='--',
        label='95% confidence'
    )
    plt.gca().add_patch(ellipse)
    
# check percentile points in the ellipse
for points, Sigma, mean in zip([egoveichle_o.samples, o_c_list, o_c_future_list], [o_e_Sigma, o_c_Sigma, o_c_future_Sigma], [o_e_mean, o_c_mean, o_c_future_mean]):
    sum = 0
    P = Sigma[:2, :2]
    w, v = np.linalg.eig(P)
    angle = np.arctan2(v[1, 0], v[0, 0])
    for point in points:
        diff = np.array([point.x - mean.x, point.y - mean.y])
        if diff @ np.linalg.inv(P) @ diff < 2.4477**2:
            sum += 1
    print(sum / len(points))

# check percentile points in the 0.95 confidence ellipse
sum = 0
P = o_c_Sigma[2, 2]
for point in o_c_list:
    diff = point.theta - o_c_mean.theta
    if diff**2 < 2**2 * P:
        sum += 1
print(sum / len(o_c_list))

sum = 0
P = o_c_future_Sigma[3, 3]
for point in o_c_list:
    diff = point.vel - o_c_mean.vel
    if diff**2 < 2**2 * P:
        sum += 1
print(sum / len(o_c_list))
    

plt.legend()
plt.grid()
plt.savefig('uncertainty_simulation.png')
plt.close()

plt.figure(figsize=(10, 4))
# ratio of x and y is 1:1
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-10, 40)
plt.ylim(-10, 10)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.scatter([e_c.x for e_c in cyclyst_e.samples], [e_c.y for e_c in cyclyst_e.samples], c='tab:orange', marker='x', alpha=0.1)
plt.scatter(0, 0, c='black', marker='*', label='ego-vehicle mean')
plt.scatter(cyclyst_e.x.mean, cyclyst_e.y.mean, c='black', marker='h', label='cyclist mean')

plt.legend()
plt.grid()
plt.savefig('uncertainty_simulation_ego.png')
plt.close()


#viorin plot of velocity and theta
plt.figure()
plt.violinplot([o_c.vel for o_c in o_c_list], showmedians=True)
plt.ylabel('velocity [m/s]')
plt.axhline(y=15/3.6+np.sqrt(o_c_Sigma[3,3]), color='r', linestyle='--')
plt.axhline(y=15/3.6-np.sqrt(o_c_Sigma[3,3]), color='r', linestyle='--', label='95% confidence')
plt.legend()
plt.savefig('velocity_violin.png')

plt.figure()
plt.violinplot([o_c.theta for o_c in o_c_list], showmedians=True)
plt.ylabel('theta [rad]')
plt.axhline(y=0+np.sqrt(o_c_Sigma[2,2]), color='r', linestyle='--')
plt.axhline(y=0-np.sqrt(o_c_Sigma[2,2]), color='r', linestyle='--', label='95% confidence')
plt.legend()
plt.savefig('theta_violin.png')



