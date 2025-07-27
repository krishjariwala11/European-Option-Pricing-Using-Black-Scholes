import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Option Pricing Calculator", layout="wide")

# Helper functions
N = lambda x: norm.cdf(x)
P = lambda x: norm.pdf(x)

class BlackScholes:
    def __init__(self, S, K, T, sigma, r):
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma / 100
        self.r = r / 100
        self.d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        self.d2 = self.d1 - self.sigma * np.sqrt(T)

    def option_price(self, type):
        if type == "call":
            return round(N(self.d1) * self.S - N(self.d2) * self.K * np.exp(-self.r * self.T), 3)
        elif type == "put":
            return round(N(-self.d2) * self.K * np.exp(-self.r * self.T) - N(-self.d1) * self.S, 3)

    def get_greeks(self, type="call"):
        delta = N(self.d1) if type == "call" else -N(-self.d1)
        gamma = P(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * P(self.d1) * np.sqrt(self.T) * 0.01
        if type == "call":
            theta = (-self.S * P(self.d1) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * N(self.d2)) / 365
            rho = self.K * self.T * np.exp(-self.r * self.T) * N(self.d2) * 0.01
        else:
            theta = (-self.S * P(self.d1) * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * N(-self.d2)) / 365
            rho = -self.K * self.T * np.exp(-self.r * self.T) * N(-self.d2) * 0.01

        return {
            "Delta": round(delta, 3),
            "Gamma": round(gamma, 3),
            "Vega": round(vega, 3),
            "Theta": round(theta, 3),
            "Rho": round(rho, 3)
        }

# Sidebar inputs
st.sidebar.header("Enter Option Parameters")
S = st.sidebar.number_input("Stock Price (S)", value=100.0, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=100.0, format="%.2f")
T = st.sidebar.number_input("Time to Expiry (in years)", value=1.0, min_value=0.01, format="%.2f")
sigma = st.sidebar.number_input("Volatility (%)", value=20.0, min_value=0.01, format="%.2f")
r = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, format="%.2f")

submit = st.sidebar.button("Calculate Option Price")

# Main layout
st.title("📈 Black-Scholes Option Pricing Calculator")
st.markdown("Visualize and understand option pricing and Greeks using the Black-Scholes model.")

with st.expander("📘 View Black-Scholes Formula"):
    st.latex(r"""
    d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad
    d_2 = d_1 - \sigma\sqrt{T}
    """)
    st.latex(r"""
    \text{Call} = S \cdot N(d_1) - K e^{-rT} N(d_2)
    """)
    st.latex(r"""
    \text{Put} = K e^{-rT} N(-d_2) - S \cdot N(-d_1)
    """)
if submit:
    bs = BlackScholes(S, K, T, sigma, r)

    call_price = bs.option_price("call")
    put_price = bs.option_price("put")

    call_greeks = bs.get_greeks("call")
    put_greeks = bs.get_greeks("put")

    # Display results in a table
    st.subheader("📊 Option Prices & Greeks")
    df = pd.DataFrame({
        "Type": ["Call", "Put"],
        "Price": [call_price, put_price],
        **{greek: [call_greeks[greek], put_greeks[greek]] for greek in call_greeks}
    })
    st.dataframe(df.set_index("Type"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    # Price vs Volatility Plot
    vol_range = np.linspace(1, 100, 50)
    call_prices_vol = [BlackScholes(S, K, T, v, r).option_price("call") for v in vol_range]
    put_prices_vol = [BlackScholes(S, K, T, v, r).option_price("put") for v in vol_range]

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(vol_range, call_prices_vol, label="Call Price", color='blue')
        ax1.plot(vol_range, put_prices_vol, label="Put Price", color='red')
        ax1.set_title("Option Price vs Volatility (Vega)")
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)

    # Price vs Time Plot
    time_range = np.linspace(0.01, 2, 50)
    call_prices_time = [BlackScholes(S, K, t, sigma, r).option_price("call") for t in time_range]
    put_prices_time = [BlackScholes(S, K, t, sigma, r).option_price("put") for t in time_range]

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(time_range, call_prices_time, label="Call Price", color='green')
        ax2.plot(time_range, put_prices_time, label="Put Price", color='orange')
        ax2.set_title("Option Price vs Time to Maturity (Theta)")
        ax2.set_xlabel("Time to Expiry (Years)")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)

    # Price vs Stock Price Plot
    stock_range = np.linspace(S * 0.5, S * 1.5, 50)
    call_prices_stock = [BlackScholes(s, K, T, sigma, r).option_price("call") for s in stock_range]
    put_prices_stock = [BlackScholes(s, K, T, sigma, r).option_price("put") for s in stock_range]

    with col3:
        fig3, ax3 = plt.subplots()
        ax3.plot(stock_range, call_prices_stock, label="Call Price", color='purple')
        ax3.plot(stock_range, put_prices_stock, label="Put Price", color='brown')
        ax3.set_title("Option Price vs Stock Price (Delta)")
        ax3.set_xlabel("Stock Price (S)")
        ax3.set_ylabel("Price")
        ax3.legend()
        st.pyplot(fig3)
