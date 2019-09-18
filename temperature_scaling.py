import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from scaling_utils import softmax


class TemperatureScaling:

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Args:
            temp (float): starting temperature, default is 1
            maxiter (int): maximum iterations done by optimizer
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver


    def _loss_function(self, x, logits, labels):
        # Calculate the loss with sklearn's log loss
        scaled_probs = self.predict(logits, x)
        loss = log_loss(y_true=labels, y_pred=scaled_probs)
        return loss

    
    # Find the temperature with scipy
    def fit(self, logits, labels):
        """
        Trains the model and finds optimal temperature

        Args:
            logits: the output from neural network for each class, shape in (num_samples, classes)
            labels: labels labels, which need to be one-hot encoded later

        Return:
            The results of optimization after loss minimizing is finished.
        """
        labels = labels.flatten()
        
        # Task 1-2: find the optimized temperature T with scipy.optimize
        ##################################


        ##################################


        print("The resulted temperature is: {:.3f}".format(self.temp))

        return optimizer

    
    def predict(self, logits, temp=None):
        """
        Scales logits with given temperature and returns calibrated probabilities

        Args:
            logits: the output from neural network for each class, shape in (num_samples, classes)
            temp: use temperatures find by optimizer or previously set.

        Return:
            Calibrated probabilities, array with shape (num_samples, classes)
        """

        # Task 1-3: predict with temperature scaling
        ##################################


        ##################################
