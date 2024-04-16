# PETER GAME
#
# This is a machine learning tool developed to predicit the direction of the market tomorrow for a defined ticker.
# The FTSE is the current tracked index
# The model is based off of:
#
# - Open Price
# - Close Ratio
# - Trend
#
# Trading horizons are defined in an array, right now as [1, 2, 7, 14, 30]
# The model is trained by taking each day that the stock hs existed in a given period and looking at the values defined above for -
# each of the trading horizons defined above. The integers in the list refer to days ago, so it checks yesterdays values (open, close ratio, trend),
# last weeks values, 2 weeks ago values, etc.... From this the model is made from considering these values anc backtesting against itself.
# Once the model has been trained through the stocks history, it is fed todays data and predicts a value for how certain it is as to what direction the -
# market is going to go tomorrow.
#
# The accuracy of this model is variable due to the random factors of the stock market, however the higheest accuracy reached was a 68.8% score of predicting -
# the FTSE ticker.


import yfinance as yf 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

ticker = yf.Ticker("^FTSE") #instanciates the ticker were using. in this case the FTSE index

ticker = ticker.history(period="max") #Set period and interval values for api call

del ticker["Dividends"]; del ticker["Stock Splits"] #Remove empty columns

ticker["Next"] = ticker["Close"].shift(-1, fill_value=10000000000) #Makes a column for next days price

#Generates a Targer column which is a boolean of whether the market went up or down. This is for the ML algorithm to use as the target

ticker["Target"] = ((ticker["Next"] > ticker["Close"]).astype(int)) 

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) #instanciates a ML model with values

horizons = [1, 2, 7, 14, 30] #Sets the comparison horizons for the ML to use as stock indicators
new_predictors = []

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"]) #fits the model with the predictors and trains it against the targets
    preds = model.predict_proba(test[predictors])[:,1] #then asks the model to predict the probability of the market increasing or decreasing the next day
    preds[preds >= 0.6] = 1 #only tells you to buy or sell if the preidictor is above 60% sure
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1) 
    return combined #returns the guesses and real values for the market movement

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy() #trains the model off of the element before 'i'
        test = data.iloc[i:(i+step)].copy() #tests against the next day
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


for horizon in horizons: #iterate through horizons generating trend ration values and close ratios
    rolling_averages = ticker.rolling(horizon).mean()
        
    ratio_column = f"Close_Ratio_{horizon}"
    ticker[ratio_column] = ticker["Close"] / rolling_averages["Close"] #calculates the close ratio fot each set trading horizon
        
    trend_column = f"Trend_{horizon}"
    ticker[trend_column] = ticker.shift(1).rolling(horizon).sum()["Target"] #calculates the trend for each set trading horizon

    new_predictors+= [ratio_column, trend_column, "Open"]

ticker = ticker.dropna() #remove empty rows

predictions = backtest(ticker, model, new_predictors) 

print(ticker)

print("===========================================")

print(predictions["Predictions"].value_counts()) #displays how often the market went up and down
print("Score: " + str(precision_score(predictions["Target"], predictions["Predictions"]))) # displays the certainty score of the model

print(ticker.iloc[-1:]) 
next_trade = model.predict_proba(ticker.iloc[-1:][new_predictors])[:,1] #takes the trained model and asks it to predict tomorrows market direction

print("Trade direction: " + str(next_trade))

if next_trade >= 0.6: #if the AI is certain enough of the market moving in either direction it tells you to buy or sell
    print("BUY")
elif next_trade < 0.5:
    print("SELL")

print("===========================================")




