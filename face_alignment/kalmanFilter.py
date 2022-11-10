import numpy as np
class extendedKalmanFilter():
    def __init__(self, A, Q, C, H, num_steps):
        self.A = A  # state transition matrix
        self.Q = Q  # covariances of driving noise & observation noise
        self.C = C  # varainces of observation errors
        self.H = H  # Observation matrix relating signal to observation
        self.num_steps = num_steps

        self.x_n = np.zeros((2, self.num_steps))

        self.estimated_signal = np.zeros((4, self.num_steps+1))
        self.estimated_MSE = np.zeros((self.num_steps+1, 4, 4))
        self.predicted_signal = np.zeros_like(self.estimated_signal)
        self.predicted_MSE = np.zeros_like(self.estimated_MSE)

        self.K = np.zeros((self.num_steps+1, 4, 2))
        # self.h = np.zeros((2, self.num_steps+1))
        self.H = np.zeros((self.num_steps+1, 2, 4))

    def initialize(self, signal_start, MSE_start):

        self.estimated_signal[:, 0] = signal_start
        self.estimated_MSE[0, :, :] = MSE_start

    def predict(self, nn):

        self.predicted_signal[:, nn] = self.A @ self.estimated_signal[:, nn-1]
        self.predicted_MSE[nn, :, :] = self.A @ self.estimated_MSE[nn-1, :, :] @ \
            self.A.transpose() + self.Q

    '''
    def calculate_hs(self, nn):

        self.h[:, nn] = np.array( (np.sqrt(self.predicted_signal[0, nn]**2 + \
                                          self.predicted_signal[1, nn]**2),
                                 atan2(self.predicted_signal[1, nn],
                                       self.predicted_signal[0, nn]) ))

    def calculate_Hn(self, nn):

        R = np.sqrt(np.sum(self.predicted_signal[:, nn]**2))

        self.H[nn, 0, 0] = self.predicted_signal[0, nn] / R
        self.H[nn, 0, 1] = self.predicted_signal[1, nn] / R
        self.H[nn, 1, 0] = -self.predicted_signal[1, nn] / (R**2)
        self.H[nn, 1, 1] = self.predicted_signal[0, nn] / (R**2)
    '''

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

    def run(self, observations):

        self.x_n = observations

        for nn in range(1, self.num_steps+1):
            self.predict(nn)
            # self.calculate_hs(nn)
            self.calculate_Hn(nn)
            self.calculate_K(nn)
            self.estimate(nn)
