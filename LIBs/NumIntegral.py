import numpy as np
import matplotlib.pyplot as plt

# 사다리꼴 방법과 심슨 방법을 포함한 수치적분 클래스
class NumericalIntegration:
    def __init__(self, f, a, b, n):
        """
        NumericalIntegration 클래스 초기화
        :param f: 적분할 함수
        :param a: 적분 구간 시작점
        :param b: 적분 구간 끝점
        :param n: 분할할 구간의 수 (심슨 방법에서는 짝수여야 함)
        """
        self.f = f  # 적분할 함수
        self.a = a  # 적분 구간 시작
        self.b = b  # 적분 구간 끝
        self.n = n  # 구간 수

    def trapezoidal_rule(self):
        """
        사다리꼴 방법으로 적분 수행
        """
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        return integral

    def simpsons_rule(self):
        """
        심슨 방법으로 적분 수행 (n은 짝수여야 함)
        """
        if self.n % 2 == 1:
            raise ValueError("n은 짝수여야 합니다.")
        
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        integral = h / 3 * (y[0] + 4 * np.sum(y[1:self.n:2]) + 2 * np.sum(y[2:self.n-1:2]) + y[self.n])
        return integral

    def display_results(self):
        """
        사다리꼴 방법과 심슨 방법의 결과 출력
        """
        trapezoidal_result = self.trapezoidal_rule()
        simpsons_result = self.simpsons_rule()
        print(f"사다리꼴 방법으로 계산한 적분값: {trapezoidal_result}")
        print(f"심슨 방법으로 계산한 적분값: {simpsons_result}")

    def plot_integration(self):
        """
        함수와 적분 구간을 시각화
        """
        x_vals = np.linspace(self.a, self.b, 1000)
        y_vals = self.f(x_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label='f(x)', color='blue')
        plt.fill_between(x_vals, y_vals, color='lightblue', alpha=0.5)
        plt.title('Numerical Integration of f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

# 사다리꼴 방법과 심슨 방법을 포함한 수치적분 클래스 (데이터 포인트 기반)
class NumericalIntegrationFromPoints:
    def __init__(self, x_values, y_values):
        """
        NumericalIntegrationFromPoints 클래스 초기화
        :param x_values: x 데이터 포인트
        :param y_values: y 데이터 포인트 (f(x) 값들)
        """
        self.x_values = np.array(x_values)
        self.y_values = np.array(y_values)
        self.n = len(x_values) - 1  # 구간의 개수

    def trapezoidal_rule(self):
        """
        사다리꼴 방법으로 적분 수행 (데이터 포인트 기반)
        """
        h = np.diff(self.x_values)  # 각 구간의 너비
        integral = np.sum(h * (self.y_values[:-1] + self.y_values[1:]) / 2)
        return integral

    def simpsons_rule(self):
        """
        심슨 방법으로 적분 수행 (n은 짝수여야 함, 데이터 포인트 기반)
        """
        if self.n % 2 == 1:
            raise ValueError("데이터 포인트의 개수가 홀수여야 심슨 방법을 사용할 수 있습니다.")
        
        h = np.diff(self.x_values)[0]  # 모든 구간의 너비가 동일하다고 가정
        integral = h / 3 * (self.y_values[0] + 4 * np.sum(self.y_values[1:self.n:2]) + 
                            2 * np.sum(self.y_values[2:self.n-1:2]) + self.y_values[self.n])
        return integral

    def display_results(self):
        """
        사다리꼴 방법과 심슨 방법의 결과 출력
        """
        trapezoidal_result = self.trapezoidal_rule()
        simpsons_result = self.simpsons_rule()
        print(f"사다리꼴 방법으로 계산한 적분값: {trapezoidal_result}")
        print(f"심슨 방법으로 계산한 적분값: {simpsons_result}")

    def plot_integration(self):
        """
        함수와 적분 구간을 시각화 (데이터 포인트 기반)
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.x_values, self.y_values, 'o-', label='Data points', color='blue')
        plt.fill_between(self.x_values, self.y_values, color='lightblue', alpha=0.5)
        plt.title('Numerical Integration from Data Points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Test function: f(x) = x^2
    def test_function(x):
        return x**2

    # Integration bounds and number of intervals
    a = 0  # Start of the interval
    b = 10  # End of the interval
    n = 10  # Number of intervals (even for Simpson's rule)

    # Create an instance of the NumericalIntegration class
    integration = NumericalIntegration(f=test_function, a=a, b=b, n=n)

    # Display results for both methods
    print("Testing NumericalIntegration:")
    integration.display_results()

    # Plot the function and the integration region
    integration.plot_integration()
    # Define data points (x, f(x))
    x_values = np.linspace(0, 10, 11)  # 11 points (10 intervals)
    y_values = x_values**2  # Corresponding y values for f(x) = x^2

    # Create an instance of the NumericalIntegrationFromPoints class
    integration_points = NumericalIntegrationFromPoints(x_values=x_values, y_values=y_values)

    # Display results for both methods
    print("\nTesting NumericalIntegrationFromPoints:")
    integration_points.display_results()

    # Plot the function and the integration region based on points
    integration_points.plot_integration()
