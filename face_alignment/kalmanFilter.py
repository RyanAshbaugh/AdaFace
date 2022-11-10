import numpy as np

class kalmanFilter():
    def __init__(self, A, Q, C, H, num_steps, initial_signal_estimate,
                 initial_MSE_estimate):
        self.A = A  # state transition matrix
        self.Q = Q  # covariances of driving noise & observation noise
        self.C = C  # varainces of observation errors
        self.H = H  # Observation matrix relating signal to observation
        self.num_steps = num_steps

        # keep list of x's
        self.x_n = np.zeros((self.H.shape[0], self.num_steps))

        self.estimated_signal = np.zeros((self.A.shape[0], self.num_steps+1))
        self.estimated_MSE = np.zeros((self.num_steps+1,
                                       *self.A.shape))
        self.predicted_signal = np.zeros_like(self.estimated_signal)
        self.predicted_MSE = np.zeros_like(self.estimated_MSE)

        self.K = np.zeros((self.num_steps+1, *self.H.shape[::-1]))
        self.Hn = np.zeros((self.num_steps+1, *self.H.shape))

        # initialize
        self.estimated_signal[:, 0] = initial_signal_estimate
        self.estimated_MSE[0, :, :] = initial_MSE_estimate

        self.prev_ii = 0
        self.missed_frames = 0

    def predict(self, nn):

        self.predicted_signal[:, nn] = self.A @ self.estimated_signal[:, nn-1]
        self.predicted_MSE[nn, :, :] = self.A @ self.estimated_MSE[nn-1, :, :] @ \
            self.A.transpose() + self.Q

    def calculate_Hn(self, nn, step_size):

        self.Hn[nn, :, :6] = step_size * np.eye(6)

    def calculate_K(self, nn):

        self.K[nn, :, :] = self.predicted_MSE[nn, :, :] @ \
            self.H[nn, :, :].transpose() @ \
            np.linalg.inv(self.C + self.H[nn, :, :] @ \
                          self.predicted_MSE[nn, :, :] @ \
                          self.H[nn, :, :].transpose())

    def estimate(self, nn):

        # adjust for x, since others need to look in the past, but nn-1 not in
        # the
        kk = nn-1
        innovation = self.x_n[:, kk] - self.h[:, nn]
        self.estimated_signal[:, nn] = self.predicted_signal[:, nn] + \
            self.K[nn, :, :] @ innovation
        self.estimated_MSE[nn, :, :] = \
            (np.eye(4) - self.K[nn, :, :] @ self.H[nn, :, :]) @ \
            self.predicted_MSE[nn, :, :]

    def predict_and_estimate(self, x, nn):

        step_size = nn - self.prev_nn
        self.x_n[:, nn] = x
        self.predict(nn)
        self.calculate_Hn(nn, step_size)
        self.calculate_K(nn)
        self.estimate(nn)
