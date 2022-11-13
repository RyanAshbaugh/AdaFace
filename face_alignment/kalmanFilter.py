import numpy as np


class kalmanFilter():
    def __init__(self,
                 num_steps,
                 sigma_rot=np.sqrt(0.1),
                 sigma_translation=np.sqrt(1)):
        '''
        self.A = A  # state transition matrix
        self.Q = Q  # covariances of driving noise & observation noise
        self.H = H
        self.C = C  # varainces of observation errors
        '''
        self.num_steps = num_steps

        # state transition matrix for warp params
        self.A = np.eye(12)
        self.A[0:6, 6:] = np.eye(6)

        # covariances of driving and observation noise
        sigma_u = np.sqrt(0.0001)
        self.Q = np.zeros((12, 12))
        self.Q[6:, 6:] = np.eye(6) * sigma_u**2

        self.sigma_rot = sigma_rot
        self.sigma_rots = self.sigma_rot**2 * np.ones(4)
        self.sigma_translation = sigma_translation
        self.sigma_translations = self.sigma_translation**2 * np.ones(2)
        sigmas_observations = np.concatenate((self.sigma_rots,
                                              self.sigma_translations))

        self.C = np.diag(sigmas_observations)

        # setup initial values for params and their derivatives
        initial_signal_estimate = np.zeros(12)
        initial_MSE_estimate = 100 * np.eye(12)

        self.H = np.zeros((6, 12))
        self.H[:6, :6] = np.eye(6)

        self.I = np.eye(12)

        # keep list of x's
        self.x_n = np.zeros((self.H.shape[0], self.num_steps))

        self.estimated_signal = np.zeros((self.A.shape[0], self.num_steps+1))
        self.estimated_MSE = np.zeros((self.num_steps+1,
                                       *self.A.shape))
        self.predicted_signal = np.zeros_like(self.estimated_signal)
        self.predicted_MSE = np.zeros_like(self.estimated_MSE)

        self.K = np.zeros((self.num_steps+1, 12, 6))

        # initialize
        self.estimated_signal[:, 0] = initial_signal_estimate
        self.estimated_MSE[0, :, :] = initial_MSE_estimate

        self.prev_ii = 0
        self.missed_frames = 0

    def predict(self, nn):

        self.predicted_signal[:, nn] = self.A @ self.estimated_signal[:, nn-1]
        self.predicted_MSE[nn, :, :] = (self.A @ self.estimated_MSE[nn-1, :, :]
                                        @ self.A.transpose() + self.Q)

    '''
    def calculate_Hn(self, nn, step_size):

        step_size = nn - self.prev_nn
        self.Hn[nn, :, :6] = step_size * np.eye(6)
    '''

    def calculate_K(self, nn):

        self.K[nn, :, :] = self.predicted_MSE[nn, :, :] @ \
            self.H.transpose() @ \
            np.linalg.inv(self.C + self.H @
                          self.predicted_MSE[nn, :, :] @
                          self.H.transpose())

    def calculate_innovation(self, nn):
        # adjust for x, since others need to look in the past, but nn-1 not in
        # the
        kk = nn-1
        self.innovation = self.x_n[:,
                                   kk] - self.H @ self.predicted_signal[:, nn]

    def estimate(self, nn):

        self.estimated_signal[:, nn] = self.predicted_signal[:, nn] + \
            self.K[nn, :, :] @ self.innovation
        self.estimated_MSE[nn, :, :] = \
            (self.I - self.K[nn, :, :] @ self.H) @ \
            self.predicted_MSE[nn, :, :]

    def predict_and_estimate(self, x, nn):

        # add 1 to nn, since initialization is 0=n-1
        self.predict(nn)

        # if no data, leave x_n 0, and gain zero which will zero out innovation
        kk = nn-1
        if x is not None:
            self.x_n[:, kk] = x
            self.calculate_K(nn)

        self.calculate_innovation(nn)
        self.estimate(nn)
