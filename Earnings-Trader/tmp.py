from data import get_active_tickers 
tickers = get_active_tickers() 
print(len(tickers), tickers[:10]) 
