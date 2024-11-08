import numpy as np


class KalmanFilter(object):
    """2d KalmanFilter
    Args:
        X: state vector; x, y, theta, velocity
        P: state covariance matrix
    """

    def __init__(self) -> None:
        self.X = np.zeros((4, 1))
        self.P = np.eye(4)
        self.X_memory = []
        self.future_X_memory = []
        self.X_pinned_memory = []
        self.P_memory = []
        self.future_P_memory = []
        self.P_pinned_memory = []
        self.Q_memory = []
        self.K_memory = []
        self.Z_memory = []
        self.ahead_times = [3.0]
        
    def pin_state(self) -> None:
        self.X_pinned = self.X
        self.P_pinned = self.P
        self.X_pinned_memory.append(self.X_pinned)
        self.P_pinned_memory.append(self.P_pinned)
    
    def predict(self, dt: float, U: np.ndarray, Q: np.ndarray) -> None:
        """Predict next state
        Args:
            dt: time step
            U: control vector; acceleration, angular velocity
            Q: control covariance matrix
            Cs: error in motion model
        """     
        self.X_memory.append(self.X)
        self.P_memory.append(self.P)
        self.Q_memory.append(Q)
        self.Z_memory.append(self.last_Z)

        F = np.array(
            [
                [1, 0, -(self.X_pinned[3, 0]) * np.sin(self.X_pinned[2, 0]) * dt, np.cos(self.X_pinned[2, 0]) * dt],
                [0, 1, (self.X_pinned[3, 0]) * np.cos(self.X_pinned[2, 0]) * dt, np.sin(self.X_pinned[2, 0]) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [0, (np.cos(self.X_pinned[2, 0]) * dt**2) / 2],
                [0, (np.sin(self.X_pinned[2, 0]) * dt**2) / 2],
                [dt, 0],
                [0, dt],
            ]
        )
        self.X = self.X_pinned + \
            np.array(
                [
                    [self.X_pinned[3, 0] * np.cos(self.X_pinned[2, 0]) * dt],
                    [self.X_pinned[3, 0] * np.sin(self.X_pinned[2, 0]) * dt],
                    [U[1, 0] * dt],
                    [U[0, 0] * dt],
                ]
            )
        self.P = F @ self.P_pinned @ F.T + B @ Q @ B.T

    def future_predict(self, dt:float, U: np.ndarray, Q: np.ndarray) -> None:
        future_Xs = []
        future_Ps = []
        for time_ahead in self.ahead_times:
            time_ahead += dt
            F = np.array(
                [
                    [1, 0, -(self.X_pinned[3, 0]) * np.sin(self.X_pinned[2, 0]) * time_ahead, np.cos(self.X_pinned[2, 0]) * time_ahead],
                    [0, 1, (self.X_pinned[3, 0]) * np.cos(self.X_pinned[2, 0]) * time_ahead, np.sin(self.X_pinned[2, 0]) * time_ahead],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array(
                [
                    [0, (np.cos(self.X_pinned[2, 0]) * time_ahead**2) / 2],
                    [0, (np.sin(self.X_pinned[2, 0]) * time_ahead**2) / 2],
                    [time_ahead, 0],
                    [0, time_ahead],
                ]
            )
            future_X = self.X_pinned + \
                np.array(
                    [
                        [self.X_pinned[3, 0] * np.cos(self.X_pinned[2, 0]) * time_ahead],
                        [self.X_pinned[3, 0] * np.sin(self.X_pinned[2, 0]) * time_ahead],
                        [U[1, 0] * time_ahead],
                        [U[0, 0] * time_ahead],
                    ]
                )
            future_P = F @ self.P_pinned @ F.T + B @ Q @ B.T
            future_Xs.append(future_X)
            future_Ps.append(future_P)
        
        self.future_X_memory.append(future_Xs)
        self.future_P_memory.append(future_Ps)

    def update(self, Z: np.ndarray, R: np.ndarray, H: np.ndarray) -> None:
        """Update state
        Args:
            Z: measurement vector
            R: measurement covariance matrix
            H: measurement matrix
        """
        if Z.shape[0] > 1:
            self.last_Z = Z
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.K_memory.append(K)
        self.X = self.X + K @ (Z - H @ self.X)
        if self.X[3, 0] < 0:
            self.X[3, 0] = 0
        self.P = (np.eye(4) - K @ H) @ self.P @ (np.eye(4) - K @ H).T + K @ R @ K.T


if __name__ == "__main__":
    ekf = KalmanFilter()
    Z = np.array([[0.5],
                  [0.5]])
    R = np.array([[0.5, 0],
                  [0, 0.5]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    ekf.update(Z, R, H)
    print(ekf.X)
    print(ekf.P)
    U = np.array([[0.5],
                  [0.5]])
    Q = np.array([[0.5, 0],
                  [0, 0.5]])
    ekf.predict(1, U, Q)
    print(ekf.X)
    print(ekf.P)
