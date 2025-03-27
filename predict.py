import json
import sys
from LinearRegressionModel import LinearRegressionModel

def main():
    model = LinearRegressionModel(learning_rate = 0.001, epochs = 5000)
    model.load_params("thetas.json")

    while True:
        try:
            km = float(input("Enter milage in positive number (km): "))
            while km < 0:
                print("Please enter a positive number")
                km = float(input("Enter milage (km): "))
            price = model.predict_price(km)
            print(f"Predicted price for {km} km of milage: {price:.2f} euros")
            break
        except ValueError:
            print("Input is not valid. Exit Program.")
            sys.exit(0)

if __name__ == "__main__":
    main()