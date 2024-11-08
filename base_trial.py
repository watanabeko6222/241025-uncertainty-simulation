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

from kalman_filter import KalmanFilter
from moving_generator import Point, TP_State, StraightLine, MovingSimulator
import numpy as np
from sensor import GNSS, WheelVel

np.random.seed(42)


simulator = MovingSimulator(
    dt=0.1,
    init_state=TP_State(x=0, y=0, theta=0, vel=(25 / 3.6)),
    moving_line=StraightLine(start=Point(0, 0), goal=Point(1000, 0)),
    desired_vel=(25 / 3.6),
    desired_accel=1,
    desired_decel=1
)
simulator.generate_state_timestamp()
filter = KalmanFilter()
filter.X[3, 0] = 25 / 3.6
filter.P[0, 0] = 4.25**2
filter.P[1, 1] = 4.25**2
filter.P[2, 2] = 0.28**2
filter.P[3, 3] = np.pi**2
gnss = GNSS(pos_std=4.25, fps=1, do_calc_vel=True, do_clac_theta=True)
wheel_vel = WheelVel(vel_std=0.28, fps=10)
infos = [gnss, wheel_vel]
time_step = 0.1

last_observed_i = 0
sigma_ww = 0.01
sigma_aa = 0.04

for i in range(len(simulator)):
    # === start t = i update === #
    now_true_state = simulator.state_list[i]
    now_true_X = np.array(
        [[now_true_state.x], [now_true_state.y], [now_true_state.theta], [now_true_state.vel]]
    )
    # sensor update
    for info in infos:
        if i % int(10 / info.fps) == 0:
            X = now_true_X.copy()
            info.observe(X, filter=filter)
            filter.pin_state()
            last_observed_i = i
    # === end t = i update === #

    # === start t = i+1 predict === #
    dt = time_step * (i - last_observed_i + 1)
    U = np.array([[np.random.normal(0, sigma_ww)], [np.random.normal(0, sigma_aa)]])
    Q = np.array([[sigma_ww**2, 0], [0, sigma_aa**2]])
    filter.predict(dt, U, Q)
    filter.future_predict(dt, U, Q)
    # === end t = i+1 predict === #

print(len(filter.X_memory))

# plot
X_memory = np.array(filter.X_memory)

plt.figure(figsize=(10, 3))
plt.gca().set_aspect("equal")
plt.scatter(X_memory[:, 0], X_memory[:, 1], s=1, label="Estimated")
plt.scatter([state.x for state in simulator.state_list], [state.y for state in simulator.state_list], label="True",s=1)
plt.scatter([z[0] for z in filter.Z_memory], [z[1] for z in filter.Z_memory], label="Observed", s=1)

for X, Sigma in zip(filter.X_memory[::10], filter.P_memory[::10]):
    P = Sigma[:2, :2]
    w, v = np.linalg.eig(P)
    angle = np.arctan2(v[1, 0], v[0, 0])
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=(X[0], X[1]),
        width=2 * np.sqrt(w[0]),
        height=2 * np.sqrt(w[1]),
        angle=np.rad2deg(angle),
        edgecolor="tab:blue",
        facecolor="none"
    )
    plt.gca().add_patch(ellipse)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.xlim(-10, 310)
plt.ylim(-10, 10)
plt.legend()
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 2), ncol=3)
plt.grid()
plt.savefig("kalman_filter.png")
plt.close()

time = np.arange(0, len(filter.X_memory) * 0.1, 0.1)

plt.figure()
# plot mahalanobis distance
mahalanobis_list = []
diff_list = []
for i, (state, X, P) in enumerate(zip(simulator.state_list, filter.X_memory, filter.P_memory)):
    diff = np.array([state.x - X[0], state.y - X[1]])
    diff_list.append(np.linalg.norm(diff))
    mahalanobis = diff.T @ np.linalg.inv(P[:2, :2]) @ diff
    mahalanobis_list.append(mahalanobis[0, 0])

plt.plot(time, mahalanobis_list, label="Mahalanobis distance")
plt.xlabel("Time [s]")
plt.ylabel("Mahalanobis distance")
plt.legend()
plt.grid()
plt.savefig("kalman_filter_mahalanobis.png")
plt.close()

plt.figure()
plt.plot(time, diff_list, label="Position error")
plt.xlabel("Time [s]")
plt.ylabel("Position error [m]")
plt.legend()
plt.grid()
plt.savefig("kalman_filter_pos_err.png")
plt.close()
    
plt.figure()
plt.plot(time, [np.sqrt(P[0, 0]) for P in filter.P_memory], label="X std")
plt.plot(time, [np.sqrt(P[1, 1]) for P in filter.P_memory], label="Y std")
plt.xlabel("Time [s]")
plt.ylabel("std [m]")
plt.legend()
plt.grid()
plt.savefig("kalman_filter_pos_std.png")
plt.close()

plt.figure()
plt.plot([np.sqrt(P[2, 2]) for P in filter.P_memory], label="Theta std")
plt.xlabel("Time [s]")
plt.ylabel("std [rad]")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("kalman_filter_theta_std.png")
plt.close()

plt.figure()
plt.plot(time, [X[2] for X in filter.X_memory], label="Theta")
# fill 2 sigma
plt.fill_between(
    time,
    [float(X[2] - 2 * np.sqrt(P[2, 2])) for X, P in zip(filter.X_memory, filter.P_memory)],
    [float(X[2] + 2 * np.sqrt(P[2, 2])) for X, P in zip(filter.X_memory, filter.P_memory)],
    alpha=0.3,
    label="2 sigma"
)
plt.xlabel("Time [s]")
plt.ylabel("Theta [rad]")
plt.legend()
plt.grid()
plt.savefig("kalman_filter_theta.png")
plt.close()

plt.figure()
plt.plot(time, [np.sqrt(P[3, 3]) for P in filter.P_memory], label="Vel std")
plt.xlabel("Time [s]")
plt.ylabel("std [m/s]")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("kalman_filter_vel_std.png")
plt.close()

plt.figure()
plt.plot(time, [X[3] for X in filter.X_memory], label="Vel")
# fill 2 sigma
plt.fill_between(
    time,
    [float(X[3] - 2 * np.sqrt(P[3, 3])) for X, P in zip(filter.X_memory, filter.P_memory)],
    [float(X[3] + 2 * np.sqrt(P[3, 3])) for X, P in zip(filter.X_memory, filter.P_memory)],
    alpha=0.3,
    label="2 sigma"
)
plt.xlabel("Time [s]")
plt.ylabel("Vel [m/s]")
plt.legend()
plt.grid()
plt.savefig("kalman_filter_vel.png")
plt.close()

# for step in range(1, len(filter.X_memory), 10):
#     print(f"Step {step}")
#     print(f"X std: {np.sqrt(filter.P_memory[step][0, 0])}")
#     print(f"Y std: {np.sqrt(filter.P_memory[step][1, 1])}")
#     print(f"Theta std: {np.sqrt(filter.P_memory[step][2, 2])}")
#     print(f"Vel std: {np.sqrt(filter.P_memory[step][3, 3])}")
#     print("")
