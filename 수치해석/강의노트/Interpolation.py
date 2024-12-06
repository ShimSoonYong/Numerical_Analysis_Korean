import numpy as np
import matplotlib.pyplot as plt

class LagrangeInterpolation:
    """
    A class for performing Lagrange polynomial interpolation.

    Attributes:
        x_values (np.array): Array of x-coordinates of the data points.
        y_values (np.array): Array of y-coordinates of the data points.

    Methods:
        interpolate(x):
            Computes the interpolated y value at a given x using the Lagrange polynomial.
        plot(x_interp_points=1000, x_test=None):
            Visualizes the interpolation along with the data points.
    """
    def __init__(self, x_values, y_values):
        """
        Initializes the LagrangeInterpolation instance with given data points.

        Args:
            x_values (list or np.array): The x-coordinates of the data points.
            y_values (list or np.array): The y-coordinates of the data points.

        Raises:
            ValueError: If the lengths of x_values and y_values do not match.
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length.")
        self.x_values = np.array(x_values)
        self.y_values = np.array(y_values)

    def interpolate(self, x):
        """
        Computes the interpolated y value at a given x using the Lagrange polynomial.

        Args:
            x (float): The x-coordinate where the interpolation is performed.

        Returns:
            float: The interpolated y value.
        """
        n = len(self.x_values)
        result = 0
        for i in range(n):
            term = self.y_values[i]
            for j in range(n):
                if i != j:
                    term *= (x - self.x_values[j]) / (self.x_values[i] - self.x_values[j])
            result += term
        return result

    def plot(self, x_interp_points=1000, x_test=None):
        """
        Visualizes the Lagrange interpolation polynomial and data points.

        Args:
            x_interp_points (int, optional): Number of points to use for plotting the interpolation curve.
                                             Defaults to 1000.
            x_test (float, optional): A specific x-coordinate to interpolate and display on the plot.
                                       Defaults to None.

        Raises:
            ValueError: If x_test is outside the range of x_values.

        Example:
            >>> x_values = [0, 1, 2]
            >>> y_values = [1, 3, 2]
            >>> interpolator = LagrangeInterpolation(x_values, y_values)
            >>> interpolator.plot(x_test=1.5)
        """
        # Ensure x_test is within range
        if x_test is not None and (x_test < min(self.x_values) or x_test > max(self.x_values)):
            raise ValueError("x_test is outside the range of x_values.")

        # Generate interpolation points
        x_interp = np.linspace(min(self.x_values), max(self.x_values), x_interp_points)
        y_interp = np.array([self.interpolate(x) for x in x_interp])

        # Plot the interpolation
        plt.figure(figsize=(8, 6))
        plt.plot(self.x_values, self.y_values, 'ro', label='Data points')  # Data points
        plt.plot(x_interp, y_interp, 'b--', label='Lagrange Interpolation')  # Interpolation curve
        
        # Plot specific interpolated point if x_test is provided
        if x_test is not None:
            y_test = self.interpolate(x_test)
            plt.scatter(x=x_test, y=y_test, s=50, c='b', label=f'Interpolated point at x={x_test}', marker='o')

        plt.title("Lagrange Polynomial Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
