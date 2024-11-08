from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point:
    x: float
    y: float


@dataclass
class TP_State:
    x: float
    y: float
    theta: float
    vel: float


@dataclass(init=False)
class StraightLine:
    start: Point
    goal: Point
    theta: float
    length: float

    def __init__(self, start: Point, goal: Point):
        self.start = start
        self.goal = goal
        self.theta = np.arctan2(goal.y - start.y, goal.x - start.x)
        self.length = np.linalg.norm(np.array([start.x, start.y]) - np.array([goal.x, goal.y]))

    def calc_point_from_length(self, length: float):
        assert length >= 0, "length should be positive"
        assert length <= self.length, "length should be smaller than line length"
        return TP_State(
            self.start.x + length * np.cos(self.theta), self.start.y + length * np.sin(self.theta), self.theta, 0
        )


class MovingSimulator:
    def __init__(
        self,
        dt: float,
        init_state: TP_State,
        moving_line: StraightLine,
        desired_vel: float,
        desired_accel: float,
        desired_decel: float,
    ):
        self.dt = dt
        self.map = map
        self.init_point = Point(init_state.x, init_state.y)
        self.init_state = init_state
        self.state_list = [init_state]
        self.moving_line = moving_line
        
        self.desired_vel = desired_vel
        self.desired_accel = desired_accel
        self.desired_decel = desired_decel

    def generate_state_timestamp(self):
        max_offset = self.moving_line.length
        now_offset = 1e-17
        last_X = self.state_list[-1]
        while now_offset + self.dt * last_X.vel < max_offset:
            now_offset += self.dt * last_X.vel
            
            now_X = self.moving_line.calc_point_from_length(now_offset)
            
            # calc accel # now in node
            if last_X.vel < self.desired_vel:
                acc_in_dt = (self.desired_vel - last_X.vel) / self.dt
                if acc_in_dt > self.desired_accel:
                    acc = self.desired_accel
                else:
                    acc = acc_in_dt
            elif last_X.vel > self.desired_vel:
                decel_in_dt = (last_X.vel - self.desired_vel) / self.dt
                if decel_in_dt > self.desired_decel:
                    acc = -self.desired_decel
                else:
                    acc = -decel_in_dt
            else:
                acc = 0
            
            now_X.vel = last_X.vel + acc * self.dt
            self.state_list.append(now_X)
            last_X = now_X

    def __len__(self):
        return len(self.state_list)
    
    def plot_xy(self):
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_title("XY")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.scatter([X.x for X in self.state_list], [X.y for X in self.state_list])
        plt.savefig("xy.png")
    
    def plot_vel(self):
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_title("Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]")
        plt.plot([self.dt * i for i in range(len(self.state_list))], [X.vel for X in self.state_list])
        plt.savefig("vel.png")


if __name__ == "__main__":

    moving_line = StraightLine(start=Point(0, 0), goal=Point(10, 0))
    simulator = MovingSimulator(
        dt=0.1,
        init_state=TP_State(x=0, y=0, theta=0, vel=0),
        moving_line=moving_line,
        desired_vel=(15 / 3.6),
        desired_accel=1,
        desired_decel=1,
    )
    simulator.generate_state_timestamp()
    
    simulator.plot_xy()
    simulator.plot_vel()
