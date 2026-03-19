import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np


def draw_plot():
    # Import data
    df = pd.read_csv("epa-sea-level.csv")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], s=10)

    # Line of best fit (1880–latest year)
    res_all = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    years_extended = np.arange(df["Year"].min(), 2051)
    sea_level_all = res_all.intercept + res_all.slope * years_extended
    ax.plot(years_extended, sea_level_all, 'r', label="1880–2050 trend")

    # Line of best fit (2000–latest year)
    df_recent = df[df["Year"] >= 2000]
    res_recent = linregress(df_recent["Year"], df_recent["CSIRO Adjusted Sea Level"])
    years_recent = np.arange(2000, 2051)
    sea_level_recent = res_recent.intercept + res_recent.slope * years_recent
    ax.plot(years_recent, sea_level_recent, 'g', label="2000–2050 trend")

    # Labels and title
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")
    ax.set_title("Rise in Sea Level")

    # Legend
    ax.legend()

    # Save and return figure
    fig.savefig("sea_level_plot.png")
    return fig