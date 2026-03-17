import requests
import pandas as pd
import matplotlib.pyplot as plt

coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana"
}

def get_price(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    data = requests.get(url).json()
    return [p[1] for p in data["prices"]]

def analyze(prices):
    change = ((prices[-1] - prices[0]) / prices[0]) * 100
    if change > 2:
        return "Bullish 📈", change
    elif change < -2:
        return "Bearish 📉", change
    else:
        return "Neutral ➡️", change

print("=== Crypto Analysis ===\n")

for name, coin in coins.items():
    print(f"===== {name} =====")

    prices_1d = get_price(coin, 1)
    prices_7d = get_price(coin, 7)

    short = analyze(prices_1d[-10:])
    day = analyze(prices_1d)
    week = analyze(prices_7d)

    print(f"30 min: {short[0]} ({short[1]:.2f}%)")
    print(f"24h: {day[0]} ({day[1]:.2f}%)")
    print(f"7d: {week[0]} ({week[1]:.2f}%)\n")

    # Chart
    plt.plot(prices_7d)
    plt.title(name + " (7 Days)")
    plt.show()
