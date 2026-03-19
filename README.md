# Sea Level Predictor

This project analyzes historical global sea level data and uses linear regression to predict future sea level rise through the year 2050. It is part of the Data Analysis with Python certification by freeCodeCamp.

The dataset used in this project contains measurements of global average sea level change from 1880 to recent years. The goal is to visualize past trends and estimate future sea level rise based on historical patterns.

The project begins by importing the dataset using the pandas library. A scatter plot is created using matplotlib, where the x-axis represents the year and the y-axis represents the sea level in inches. This provides a visual representation of how sea levels have changed over time.

Next, a line of best fit is calculated using linear regression from the scipy.stats module. The first regression line is based on the entire dataset (from 1880 onward) and is extended to the year 2050 to predict long-term trends in sea level rise.

A second regression line is then created using only the data from the year 2000 to the most recent available year. This provides a more recent trend, which may better reflect the current rate of sea level rise. This line is also extended to 2050 for comparison.

The final plot includes both regression lines along with the original data points. Proper labels and a title are added to clearly communicate the information. The resulting visualization is saved as an image file.

This project demonstrates how data analysis techniques such as data visualization and linear regression can be used to understand trends and make predictions about real-world environmental changes.

Technologies used in this project include Python, pandas, matplotlib, scipy, and numpy.
