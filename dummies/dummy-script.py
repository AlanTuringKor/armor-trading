import ccxt
import time

def binance_fetch():
    # Create Binance instance
    binance = ccxt.binance()

    # Specify the trading pair
    symbol = 'BTC/KRW'  # Replace with the desired trading pair 
    # Fetch ticker data
    ticker = binance.fetch_ticker(symbol)

    # Extract relevant information
    timestamp = ticker['timestamp']
    open_price = ticker['open']
    high_price = ticker['high']
    low_price = ticker['low']
    close_price = ticker['close']
    volume = ticker['quoteVolume']

    # Print or process the data as needed
    print(f'Binance => Timestamp: {timestamp}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}')




def upbit_fetch():
    # Replace with your actual Upbit API key and secret
    api_key = '4p0xmJE5BbcxYnakY4R5nwdFgWbUShgwJabYxLio'
    api_secret = 'II33U06p4vZH0AGBxDGSckWC5pr5x9ThZDLPNjh1'

    # Create Upbit instance
    upbit = ccxt.upbit({
        'apiKey': api_key,
        'secret': api_secret,
    })  

    # Specify the trading pair
    symbol = 'BTC/KRW'  # Replace with the desired trading pair

    try:
        # Fetch ticker data
        ticker = upbit.fetch_ticker(symbol)

        # Extract relevant information
        timestamp = ticker['timestamp']
        open_price = ticker['open']
        high_price = ticker['high']
        low_price = ticker['low']
        close_price = ticker['close']
        volume = ticker['quoteVolume']

        # Print or process the data as needed
        print(f'Upbit => Timestamp: {timestamp}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}')

        # Wait for one minute before fetching the next data


    except ccxt.NetworkError as e:
        print(f'Network error: {e}')
        time.sleep(10)

    except ccxt.ExchangeError as e:
        print(f'Exchange error: {e}')
        time.sleep(10)

    except Exception as e:
        print(f'Error: {e}')
        time.sleep(10)


# Main loop to fetch data every minute
while True:
    upbit_fetch()
    #binance_fetch()
    time.sleep(5)
