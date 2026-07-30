The purpose of this project was to create an automated trading strategy and backtest it on anonymous OHLCV data. Testing and Evaluation of the developed strategy is done through BackTrader on python. 
An initial capital of £1,000,000 is also provided for simulated trading with the purpose of developing a strategy for maximising profits and minimising risks.  

The strategy combines Trend Following and Mean Reversion using Bayesian Machine Learning. A Bayesian Regime detector classifies the market per instrument in the data series into one of Trending Up/ Sideways/ Trending Down. 
The strategy is divided into 4 lookbacks(sub-strategies) and using Bayesian Model Selection and Dirichlet weighting, capital is allocated to the best performing sub-strategy. Entry Signals and Regime Detection decide when to enter a position on an instrument. There are 4 choices for taking a position - Trend Following Long/Short and Mean Reversion Long/Short.

Testing Summary final 2 strategies developed in this project across 15 data series across a varied time period of 3-6 years (Depending on Data series - check Data_Mapping)
<img width="474" height="668" alt="Screenshot 2026-07-30 at 15 17 47" src="https://github.com/user-attachments/assets/71ddd311-9d71-4439-bd5f-8026c63033bf" />
