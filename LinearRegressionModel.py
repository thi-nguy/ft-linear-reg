import math
import csv
import sys
import json
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = 0
        self.theta1 = 0
        self.x_mean = 0
        self.y_mean = 0
        self.x_std = 1
        self.y_std = 1
        
    
    def zscore_normalize(self, data):
        mean = sum(data) / len(data)
        std_deviation = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
        normalized_data = [(x - mean) / std_deviation for x in data]
        print("Data after Z-score normalization:")
        return normalized_data, mean, std_deviation
    
    def load_data(self, file_path):
        x = []
        y = []
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    x.append(float(row['km']))
                    y.append(float(row['price']))
                return x, y
        except FileNotFoundError:
            print(f"File {file_path} does not exist")
            sys.exit(1)
        except ValueError:
            print(f"Invalid data format in file {file_path} (non-numeric values)")
            sys.exit(1)
        except KeyError:
            print(f"Missing 'km' or 'price' column in file {file_path}")
            sys.exit(1)
        return x, y

    def predict_price(self, km):
        km_norm = (km - self.x_mean) / self.x_std
        price_norm = self.theta0 + self.theta1 * km_norm
        price = price_norm * self.y_std + self.y_mean
        return price
    
    def gradient_descent(self, x, y):
        m = len(x)
        for epoch in range(self.epochs):
            y_predict = [self.theta0 + self.theta1 * xi for xi in x]
            gradient_theta0 = (1/m) * sum(y_predict[i] - y[i] for i in range(m))
            gradient_theta1 = (1/m) * sum((y_predict[i] - y[i]) * x[i] for i in range(m))
            self.theta0 -= self.learning_rate * gradient_theta0
            self.theta1 -= self.learning_rate * gradient_theta1
            if epoch % 100 == 0:
                mse = self.compute_mse(y, y_predict)
                print(f"Epoch {epoch}: MSE = {mse:.4f}, theta0 = {self.theta0:.4f}, theta1 = {self.theta1:.4f}")


    def train(self, file_path):
        x, y = self.load_data(file_path)
        x_norm, self.x_mean, self.x_std = self.zscore_normalize(x)
        y_norm, self.y_mean, self.y_std = self.zscore_normalize(y)
        self.gradient_descent(x_norm, y_norm)
        print("Coefficients after training: ")
        print(f'theta0 = {self.theta0}')
        print(f'theta1 = {self.theta1}')
        params = {
            'theta0': self.theta0,
            'theta1': self.theta1,
            'x_mean': self.x_mean,
            'y_mean': self.y_mean,
            'x_std': self.x_std,
            'y_std': self.y_std
        }
        with open("thetas.json", "w") as theta_file:
            json.dump(params, theta_file)
        return x, y
    
    def plot_model(self, x, y):
        x_range = [min(x), max(x)]
        y_pred = [self.predict_price(km) for km in x_range]
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x_range, y_pred, color='red', label='Fitted line')
        plt.xlabel('Kilometers (km)')
        plt.ylabel('Price')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_mse(self, x, y, model):
        y_predict = self.predict(x)
        mse = self.compute_mse(y, y_predict)
        plt.plot([i for i in range(len(epochs))], mse, "r", color="green")
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Lost function over training time')
        plt.legend()
        plt.show()

    def compute_mse(self, y_true, y_pred):
        """Mean Squared Error"""
        m = len(y_true)
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(m)) / m

    def load_params(self, file_path):
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
                self.theta0 = params['theta0']
                self.theta1 = params['theta1']
                self.x_mean = params['x_mean']
                self.y_mean = params['y_mean']
                self.x_std = params['x_std']
                self.y_std = params['y_std']
        except FileNotFoundError:
            print("Can't find coefficents. Maybe model has not been trained yet. Using theta0 = 0, theta1 = 0")
        except (json.JSONDecodeError, KeyError):
            print("JSON file is not valid, using theta0 = 0, theta1 = 0")


def main():
        model = LinearRegressionModel(learning_rate = 0.001, epochs = 5000)
        x, y = model.train('data.csv')
        model.plot_model(x, y)

if __name__ == "__main__":
    main()    