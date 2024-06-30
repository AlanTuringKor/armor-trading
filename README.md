# Armor Trading

This project focuses on predicting and analyzing cryptocurrency prices and related metrics using machine learning techniques, specifically LSTM (Long Short-Term Memory) neural networks.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

This project utilizes historical cryptocurrency data to predict future prices and other metrics such as volume, hashrate, and transaction count. It employs LSTM neural networks for time series forecasting and provides visualization tools for analyzing the results.

## Features

- Data retrieval from various financial APIs
- Data preprocessing and scaling
- LSTM model creation and training
- Price prediction for cryptocurrencies
- Visualization of actual vs predicted values
- Analysis of prediction accuracy

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- pandas-datareader
- yfinance
- TensorFlow
- scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/AlanTuringKor/armor-trading.git
   cd armor-trading
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Set up your desired parameters in the main script:
   ```python
   start_date = dt.datetime(2021, 1, 1)
   end_date = dt.datetime(2023, 12, 24)

   test_data = MyData(start_date, end_date, 
                      [Feature.CLOSING, Feature.VOLUME, Feature.HASHRATE, Feature.TRANSACTIONCOUNT],
                      [Feature.CLOSING, Feature.VOLUME, Feature.HASHRATE, Feature.TRANSACTIONCOUNT])

   test_model = MyModel(ModelType.LSTM,[64,64,64,64,64],test_data)
   ```

2. Run the main script:
   ```
   python prediction_test.py
   ```

3. The script will train the model, make predictions, and display various visualizations and metrics.

## Project Structure

- `main.py`: The main script to run the prediction and analysis
- `model.py`: Contains the `MyModel` class for creating and training the LSTM model
- `data.py`: Contains the `MyData` class for data retrieval and preprocessing
- `my_enum.py`: Defines enums for features and model types
- `requirements.txt`: List of required Python packages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This README provides a basic structure and information about your project. You may want to expand on certain sections, add more detailed usage instructions, or include information about how to contribute to the project. Also, make sure to create a `requirements.txt` file listing all the necessary Python packages for your project.
