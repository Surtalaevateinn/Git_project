import numpy as np
import pandas as pd


class gradient:
    def __init__(self, input, output, theta1, theta0):
        self.input = input
        self.output = output
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = 0.001
        self.predict = np.zeros(len(self.input))

    def calculate(self):
        bias = self.theta0 * np.ones(len(self.input))
        self.predict = self.theta1 * self.input + self.theta0 * bias

    def cost_function(self):
        J = 0
        for i in range(len(self.input)):
            J += pow(self.predict[i] - self.output[i], 2)
        J = J / 2 / len(self.input)
        return J

    def update(self):
        sum0 = 0
        sum1 = 0
        for i in range(len(self.input)):
            sum0 += np.sum(self.predict - self.output)
            sum1 += np.sum((self.predict - self.output) * self.input)
        temp0 = self.theta0 - self.alpha / len(self.input) * sum0
        temp1 = self.theta1 - self.alpha / len(self.input) * sum1
        self.theta0 = temp0
        self.theta1 = temp1

    def run(self):
        self.calculate()
        cost = self.cost_function()
        self.update()
        print(cost)
        print(self.predict)
        print(self.theta1, self.theta0)


if __name__ == "__main__":
    input = np.array([2, 4, 6, 8])
    output = np.array([2, 5, 6, 9])
    theta0 = 0
    theta1 = 1
    gra = gradient(input, output, theta1, theta0)
    for i in range(1000):
        gra.run()
    gra1 = gradient(input, output, 1.1, 0)
    gra1.calculate()
    J = gra1.cost_function()
    print(J)


