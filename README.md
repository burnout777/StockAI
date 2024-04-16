#StockAI

Python ML model to predict the direction of tomorrows market

#PETER GAME

This is a machine learning tool developed to predicit the direction of the market tomorrow for a defined ticker.
The FTSE is the current tracked index
The model is based off of:

- Open Price
- Close Ratio
- Trend

Trading horizons are defined in an array, right now as [1, 2, 7, 14, 30]
The model is trained by taking each day that the stock hs existed in a given period and looking at the values defined above for 
each of the trading horizons defined above. The integers in the list refer to days ago, so it checks yesterdays values (open, close ratio, trend),
last weeks values, 2 weeks ago values, etc.... From this the model is made from considering these values anc backtesting against itself.
Once the model has been trained through the stocks history, it is fed todays data and predicts a value for how certain it is as to what direction the -
market is going to go tomorrow.

The accuracy of this model is variable due to the random factors of the stock market, however the higheest accuracy reached was a 68.8% score of predicting -
the FTSE ticker.
