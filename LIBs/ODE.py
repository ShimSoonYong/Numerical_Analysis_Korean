import numpy as np
import matplotlib.pyplot as plt

class ODESolver:
    def __init__(self):
        """Initialize the ODESolver class."""
        pass

    @staticmethod
    def EulerMethod(func, y0, t0, t_end, h):
        """
        Generalized Euler Method for solving ODEs.

        Parameters:
            func (callable): Function defining dy/dt = f(y, t).
            y0 (float or array-like): Initial value(s).
            t0 (float): Initial time.
            t_end (float): End time.
            h (float): Step size.

        Returns:
            tuple:
                - t_vals (numpy.ndarray): Array of time values.
                - y_vals (numpy.ndarray): Array of solution values.
        """
        if h <= 0:
            raise ValueError("Step size `h` must be positive.")
        if t0 >= t_end:
            raise ValueError("`t0` must be less than `t_end`.")

        t_vals = np.arange(t0, t_end + h, h)
        y_vals = [y0]

        for t in t_vals[:-1]:
            y_curr = y_vals[-1]
            y_next = y_curr + h * func(y_curr, t)
            y_vals.append(y_next)

        return t_vals, np.array(y_vals)

    @staticmethod
    def RK2(func, y0, t0, t_end, n):
        """
        Second-order Runge-Kutta (RK2) method for solving ODEs.

        Parameters:
            func (callable): Function dy/dt = f(y, t).
            y0 (float): Initial value y(t0).
            t0 (float): Initial time.
            t_end (float): End time.
            n (int): Number of time points.

        Returns:
            tuple:
                - t_vals (numpy.ndarray): Array of time values.
                - y_vals (numpy.ndarray): Array of solution values.
        """
        t_vals = np.linspace(t0, t_end, n)
        h = (t_end - t0) / (n - 1)
        y_vals = [y0]

        y = y0
        for t in t_vals[:-1]:
            k1 = y + (h/2)*func(t, y)
            y = y + h * func(t + (h/2), k1)
            y_vals.append(y)

        return t_vals, np.array(y_vals)
    
    @staticmethod
    def RK2_2nd(dfz, dfy, y0, z0, t0, t_end, n):
        """
        Second-order Runge-Kutta (RK2) method for solving second-order ODEs.

        Parameters:
        - dfz (callable): Function dz/dt = f(y, z, t).
        - dfy (callable): Function dy/dt = z.
        - y0 (float): Initial value y(t0).
        - z0 (float): Initial value z(t0) = dy(t0)/dt.
        - t0 (float): Initial time.
        - t_end (float): End time.
        - n (int): Number of points.

        Returns:
        - t_values (numpy.ndarray): Time points.
        - z_values (numpy.ndarray): Solution for z at each time point.
        - y_values (numpy.ndarray): Solution for y at each time point.
        """
        t_values = np.linspace(t0, t_end, n)
        y_values = [y0]
        z_values = [z0]
        h = (t_end - t0) / n  # Step size

        y = y0
        z = z0
        for t in t_values[:-1]:
            zk1 = z + (h / 2) * dfz(y, z, t)
            yk1 = y + (h / 2) * dfy(y, z, t)
            z = z + h * dfz(yk1, zk1, t + (h / 2))
            y = y + h * dfy(yk1, zk1, t + (h / 2))
            z_values.append(z)
            y_values.append(y)

        return t_values, np.array(y_values), np.array(z_values)
    
    @staticmethod
    def RK4_2nd(dfz, dfy, y0, z0, t0, t_end, n):
        """
        Fourth-order Runge-Kutta (RK4) method for solving second-order ODEs.

        Parameters:
            dfz (callable): Function dz/dt = f(y, z, t).
            dfy (callable): Function dy/dt = z.
            y0 (float): Initial value y(t0).
            z0 (float): Initial value z(t0) = dy(t0)/dt.
            t0 (float): Initial time.
            t_end (float): End time.
            n (int): Number of time points.

        Returns:
            tuple:
                - t_vals (numpy.ndarray): Array of time values.
                - y_vals (numpy.ndarray): Array of solution values for y.
                - z_vals (numpy.ndarray): Array of solution values for z.
        """
        t_vals = np.linspace(t0, t_end, n)
        h = (t_end - t0) / (n - 1)

        y_vals = [y0]
        z_vals = [z0]

        y = y0
        z = z0

        for t in t_vals[:-1]:
            # Compute k values for z and y
            k1y = h * dfy(y, z, t)
            k1z = h * dfz(y, z, t)

            k2y = h * dfy(y + 0.5 * k1y, z + 0.5 * k1z, t + 0.5 * h)
            k2z = h * dfz(y + 0.5 * k1y, z + 0.5 * k1z, t + 0.5 * h)

            k3y = h * dfy(y + 0.5 * k2y, z + 0.5 * k2z, t + 0.5 * h)
            k3z = h * dfz(y + 0.5 * k2y, z + 0.5 * k2z, t + 0.5 * h)

            k4y = h * dfy(y + k3y, z + k3z, t + h)
            k4z = h * dfz(y + k3y, z + k3z, t + h)

            # Update y and z
            y += (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
            z += (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)

            # Append the updated values
            y_vals.append(y)
            z_vals.append(z)

        return t_vals, np.array(y_vals), np.array(z_vals)


    @staticmethod
    def RK4(func, y0, t0, t_end, n):
        """
        Fourth-order Runge-Kutta (RK4) method for solving ODEs.

        Parameters:
            func (callable): Function dy/dt = f(y, t).
            y0 (float): Initial value y(t0).
            t0 (float): Initial time.
            t_end (float): End time.
            n (int): Number of time points.

        Returns:
            tuple:
                - t_vals (numpy.ndarray): Array of time values.
                - y_vals (numpy.ndarray): Array of solution values.
        """
        t_vals = np.linspace(t0, t_end, n)
        h = (t_end - t0) / (n - 1)
        y_vals = [y0]

        y = y0
        for t in t_vals[:-1]:
            k1 = func(y, t)
            k2 = func(y + 0.5 * h * k1, t + 0.5 * h)
            k3 = func(y + 0.5 * h * k2, t + 0.5 * h)
            k4 = func(y + h * k3, t + h)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_vals.append(y)

        return t_vals, np.array(y_vals)

    @staticmethod
    def shooting_method(
        dfz, dfy, y0, yn, U0, U1, x0, xn, epsil=1e-6, n=100, max_iter=1000, solver="RK4"
    ):
        """
        Shooting Method to solve boundary value problems with solvers for ODEs.

        Parameters:
            dfz (callable): Function dz/dt = f(y, z, t).
            dfy (callable): Function dy/dt = z.
            y0 (float): Initial boundary condition y(x0).
            yn (float): Boundary condition y(xn).
            U0 (float): Initial guess for z(x0) (slope).
            U1 (float): Second guess for z(x0) (slope).
            x0, xn (float): Start and end points of x.
            epsil (float): Convergence criterion for residual.
            n (int): Number of points.
            max_iter (int): Maximum number of iterations.
            solver (str): Solver method ("Euler", "RK2", "RK4").

        Returns:
            tuple:
                - Final value of U (initial slope).
                - List of (x, y) pairs.
        """
        # Step size and discretized x values
        x_values = np.linspace(x0, xn, n, endpoint=True)

        # Select the solver method
        if solver == "Euler":
            solve_method = ODESolver.EulerMethod_2nd
        elif solver == "RK2":
            solve_method = ODESolver.RK2_2nd
        elif solver == "RK4":
            solve_method = ODESolver.RK4_2nd
        else:
            raise ValueError(f"Unknown solver method: {solver}")

        def compute_residual(U_guess):
            """
            Compute the residual for a given initial slope guess.
            """
            _, y_vals, _ = solve_method(dfz, dfy, y0, U_guess, x0, xn, n)
            return y_vals[-1] - yn  # Difference between computed and target boundary

        # Initial boundary residuals
        R0 = compute_residual(U0)
        R1 = compute_residual(U1)

        if abs(R0) < epsil:
            print("Initial guess U0 satisfies the boundary condition.")
            return U0, solve_method(dfz, dfy, y0, U0, x0, xn, n)

        if abs(R1) < epsil:
            print("Second guess U1 satisfies the boundary condition.")
            return U1, solve_method(dfz, dfy, y0, U1, x0, xn, n)

        # Iterative refinement of U using the secant method
        for iter_count in range(max_iter):
            # Compute the new guess for U using the secant method
            if R1 - R0 == 0:
                raise ValueError("Division by zero in the secant method. Try different initial guesses.")
            
            U_new = U1 - R1 * (U1 - U0) / (R1 - R0)

            # Update residuals
            R_new = compute_residual(U_new)

            # Check for convergence
            if abs(R_new) < epsil:
                print(f"Converged after {iter_count + 1} iterations.")
                t_vals, y_vals, z_vals = solve_method(dfz, dfy, y0, U_new, x0, xn, n)
                return U_new, list(zip(t_vals, y_vals))

            # Update guesses
            U0, R0 = U1, R1
            U1, R1 = U_new, R_new

        raise RuntimeError(f"Failed to converge after {max_iter} iterations.")


    @staticmethod
    def Matrix_Method(A_func, b_func, delt, x0, x_end):
        """
        Solve a tridiagonal matrix system using Thomas Algorithm (TDMA).

        Parameters:
            A_func (callable): Function to calculate diagonal elements.
            b_func (callable): Function to calculate right-hand side values.
            delt (float): Step size for discretization.
            x0 (float): Start of the domain.
            x_end (float): End of the domain.

        Returns:
            tuple:
                - n (int): Number of intervals.
                - logs (list): Logs from the TDMA process.
                - solution (numpy.ndarray): Solution vector for the tridiagonal system.
        """
        if delt <= 0:
            raise ValueError("Step size `delt` must be positive.")
        if x0 >= x_end:
            raise ValueError("`x0` must be less than `x_end`.")

        n = int((x_end - x0) / delt) + 1
        a = np.full(n - 2, 1.0)  # Subdiagonal
        b = np.full(n - 1, A_func(delt))  # Main diagonal
        c = np.full(n - 2, 1.0)  # Superdiagonal
        r = np.full(n - 1, b_func(delt))  # Right-hand side

        logs = []
        for i in range(1, n - 1):
            factor = a[i - 1] / b[i - 1]
            b[i] -= factor * c[i - 1]
            r[i] -= factor * r[i - 1]
            logs.append(f"Forward Elimination Step {i}: b={b}, r={r}")

        solution = np.zeros(n - 1)
        solution[-1] = r[-1] / b[-1]
        for i in range(n - 3, -1, -1):
            solution[i] = (r[i] - c[i] * solution[i + 1]) / b[i]
            logs.append(f"Back Substitution Step {i}: solution={solution}")

        return n, logs, solution
    
    @staticmethod
    def EulerMethod_2nd(dfz, dfy, y0, z0, t0, t_end, n):
        """
        Euler Method for solving second-order ODEs.

        Parameters:
            dfz (callable): Function dz/dt = f(y, z, t).
            dfy (callable): Function dy/dt = z.
            y0 (float): Initial value y(t0).
            z0 (float): Initial value z(t0) = dy(t0)/dt.
            t0 (float): Initial time.
            t_end (float): End time.
            n (int): Number of points.

        Returns:
            tuple:
                - t_vals (numpy.ndarray): Array of time values.
                - y_vals (numpy.ndarray): Array of solution values for y.
                - z_vals (numpy.ndarray): Array of solution values for z.
        """
        if n <= 1:
            raise ValueError("Number of points `n` must be greater than 1.")
        if t0 >= t_end:
            raise ValueError("`t0` must be less than `t_end`.")

        # Step size and time values
        h = (t_end - t0) / (n - 1)
        t_vals = np.linspace(t0, t_end, n)

        # Initialize solution arrays
        y_vals = [y0]
        z_vals = [z0]

        y = y0
        z = z0

        for t in t_vals[:-1]:
            # Update y and z using Euler's method
            y_next = y + h * dfy(y, z, t)
            z_next = z + h * dfz(y, z, t)

            # Append results
            y_vals.append(y_next)
            z_vals.append(z_next)

            # Prepare for next iteration
            y, z = y_next, z_next

        return t_vals, np.array(y_vals), np.array(z_vals)

    @staticmethod
    def shooting_method(
        dfz, dfy, y0, yn, U0, U1, x0, xn, epsil=1e-6, n=100, max_iter=100, solver="RK4"
    ):
        """
        Shooting method to solve boundary value problems using Euler's method or other solvers.

        Parameters:
            dfz (callable): Function dz/dt = f(y, z, t).
            dfy (callable): Function dy/dt = z.
            y0 (float): Initial boundary value y(x0).
            yn (float): Final boundary value y(xn).
            U0 (float): Initial guess for dy/dx at x0.
            U1 (float): Second guess for dy/dx at x0.
            x0 (float): Start of the interval.
            xn (float): End of the interval.
            epsil (float): Convergence criterion for residual.
            n (int): Number of time steps.
            max_iter (int): Maximum number of iterations.
            solver (str): Solver method ("Euler", "RK2", "RK4").

        Returns:
            tuple:
                - Final slope U.
                - List of tuples [(x, y)] representing the solution.
        """
        # Step size
        delt = (xn - x0) / n

        # Select the solver method
        if solver == "Euler":
            solve_method = ODESolver.EulerMethod_2nd
        elif solver == "RK2":
            solve_method = ODESolver.RK2_2nd
        elif solver == "RK4":
            solve_method = ODESolver.RK4_2nd
        else:
            raise ValueError(f"Unknown solver method: {solver}")

        def compute_residual(U_guess):
            """
            Compute the residual R(U) = y(xn) - yn for a given slope guess.
            """
            _, y_vals, _ = solve_method(dfz, dfy, y0, U_guess, x0, xn, n)
            return y_vals[-1] - yn, y_vals

        # Compute residuals for initial guesses
        R0, y_vals0 = compute_residual(U0)
        R1, y_vals1 = compute_residual(U1)

        # Check if initial guesses satisfy the boundary condition
        if abs(R0) < epsil:
            return U0, list(zip(np.linspace(x0, xn, n + 1), y_vals0))
        if abs(R1) < epsil:
            return U1, list(zip(np.linspace(x0, xn, n + 1), y_vals1))

        # Iterative refinement using the secant method
        for _ in range(max_iter):
            if R1 == R0:
                raise ValueError("Residuals R0 and R1 are identical; cannot proceed with secant method.")

            # Update U using the secant method
            U_new = U1 - R1 * (U1 - U0) / (R1 - R0)

            # Compute new residual
            R_new, y_vals_new = compute_residual(U_new)

            # Check for convergence
            if abs(R_new) < epsil:
                print(f"Converged after {_ + 1} iterations.")
                return U_new, list(zip(np.linspace(x0, xn, n + 1), y_vals_new))

            # Update guesses
            U0, R0 = U1, R1
            U1, R1 = U_new, R_new

        # Raise an error if the method fails to converge
        raise RuntimeError(f"Failed to converge after {max_iter} iterations.")


def test_all_methods():
    # Define a simple ODE dy/dt = -2y, y(0) = 1 (solution: y(t) = exp(-2t))
    def func(y, t):
        return -2 * y

    # Define a second-order ODE system: dz/dt = -y, dy/dt = z
    def dfz(y, z, t):
        return -y

    def dfy(y, z, t):
        return z

    # Define functions for the tridiagonal matrix problem
    def A_func(delta):
        return 2 + delta

    def b_func(delta):
        return 3 * delta

    # Test parameters
    y0 = 1.0
    z0 = 0.0
    t0, t_end = 0, 5
    n_points = 50
    step_size = 0.1

    print("Testing Euler Method...")
    t_vals, y_vals = ODESolver.EulerMethod(func, y0, t0, t_end, step_size)
    plt.plot(t_vals, y_vals, label="Euler Method")
    plt.plot(t_vals, np.exp(-2 * t_vals), label="Exact Solution", linestyle="dashed")
    plt.legend()
    plt.title("Euler Method")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid()
    plt.show()

    print("Testing Euler Method for second-order ODE...")
    t_vals, y_vals, z_vals = ODESolver.EulerMethod_2nd(dfz, dfy, y0, z0, t0, t_end, step_size)
    plt.plot(t_vals, y_vals, label="y(t) (Euler 2nd-order)")
    plt.plot(t_vals, z_vals, label="z(t) = dy/dt (Euler 2nd-order)")
    plt.legend()
    plt.title("Euler Method for Second-Order ODE")
    plt.xlabel("t")
    plt.ylabel("y(t) / z(t)")
    plt.grid()
    plt.show()

    print("Testing RK2...")
    t_vals, y_vals = ODESolver.RK2(func, y0, t0, t_end, n_points)
    plt.plot(t_vals, y_vals, label="RK2 Method")
    plt.plot(t_vals, np.exp(-2 * t_vals), label="Exact Solution", linestyle="dashed")
    plt.legend()
    plt.title("RK2 Method")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid()
    plt.show()

    print("Testing RK4...")
    t_vals, y_vals = ODESolver.RK4(func, y0, t0, t_end, n_points)
    plt.plot(t_vals, y_vals, label="RK4 Method")
    plt.plot(t_vals, np.exp(-2 * t_vals), label="Exact Solution", linestyle="dashed")
    plt.legend()
    plt.title("RK4 Method")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid()
    plt.show()

    print("Testing RK4 for second-order ODE...")
    t_vals, z_vals, y_vals = ODESolver.RK4_2nd(dfz, dfy, y0, z0, t0, t_end, n_points)
    plt.plot(t_vals, y_vals, label="y(t) (RK4 2nd-order)")
    plt.plot(t_vals, z_vals, label="z(t) = dy/dt (RK4 2nd-order)")
    plt.legend()
    plt.title("RK4 Method for Second-Order ODE")
    plt.xlabel("t")
    plt.ylabel("y(t) / z(t)")
    plt.grid()
    plt.show()

    print("Testing Matrix Method...")
    x0, x_end = 0, 1
    n, logs, solution = ODESolver.Matrix_Method(A_func, b_func, step_size, x0, x_end)
    print(f"Matrix solution: {solution}")
    print(f"Number of intervals: {n}")
    print("Logs from TDMA:")
    for log in logs[-3:]:  # Print the last three logs for brevity
        print(log)

# Execute the tests
if __name__ == "__main__":
    test_all_methods()
