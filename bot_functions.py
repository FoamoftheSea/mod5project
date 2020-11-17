from bs4 import BeautifulSoup
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tda import auth, client
import time

def start_bot(keys):
    """
    Starts TD Ameritrade Scraping Bot. Takes input of dictionary containing 
    username and password which must have keys "user" and "pass" with the 
    values to be used. Returns webdriver object to be used for session.

    :param keys: (dict) dictionary with username ("user") and password ("pass")
    """
    driver = webdriver.Chrome()
    #driver.implicitly_wait(20)
    login_url = 'https://secure.tdameritrade.com/auth'
    try:
        driver.get(login_url)
    except:
        raise ValueError("Something went wrong")
    else:
        assert "TD Ameritrade" in driver.title
        username = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_css_selector('input#username'))
        username.send_keys(keys["user"])
        password = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_id("password"))
        password.send_keys(keys["pass"])
        try:
            driver.find_element_by_css_selector('input#accept.accept.button').click()
        except:
            raise ValueError("Something went wrong")
        else:
            WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Use desktop website'))
            time.sleep(3)
            button = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, value='//*[@id="app"]/div/div[2]/footer/div/ul/li[1]/button'))
            button.click()
            time.sleep(3)
            home_url = driver.current_url
            reduce_tabs(driver)

    return driver

def start_client(keys_path, token_path, redirect_url='https://localhost'):
    """
    (Deprecated): Instead make client externally and pass into functions as
    demonstrated in the notebooks.
    
    This function creates a client for API through tda. For now I recommend
    just making one externally, this is only here for if I turn the scraper
    into a class object at some point.
    """
    api_key = get_keys(keys_path)["consumer_key"]
    c = auth.easy_client(api_key, redirect_url, token_path)

    return c

def build_big_df(tickers, database_path):
    """
    This function reads a previously scraped watchlist database at the provided
    path, and combines all of the 'combined.csv' files into one dataframe.

    :param tickers: (list-like) The securities to be gathered
    :param database_path: (str) The location of the database
    """
    big_df = pd.DataFrame()
    for ticker in tickers:
        file_path = database_path+'/{}/combined.csv'.format(ticker)
        try:
            temp = pd.read_csv(file_path, index_col='Unnamed: 0').T
        except:
            temp = pd.DataFrame(pd.read_csv(file_path)).T
        big_df = pd.concat([big_df, temp.astype('float64',errors='ignore')], axis=0, sort=True)
    new_df = pd.DataFrame()
    for col in big_df:
        new_df[col] = big_df[col].astype('float64', copy=True, errors='ignore')
    for col in new_df.columns:
        if col.endswith('since'):
            new_df[col] = pd.to_datetime(new_df[col], infer_datetime_format=True)
    
    return new_df

def calculate_intrinsic(c, ticker, root_dir, method, projection_period='5yr', verbose=False,
                        pe_growth='estimate', beta=None, rfr=None, market_return=None, 
                        shares_out=None, price=None, eps=None):
    """
    This function calculates the intrinsic value of a stock based on the growth metric
    specified ('FCF', 'PB', 'PE', 'PR'), based on the data found at 'root_dir', which
    has been scraped using the functions in this package.

    :param c: (tda client) client object for session
    :param ticker: (str) the ticker symbol for the company of interest
    :param root_dir: (str) the root directory of the scraped watchlist
    :param method: (str) growth metric to use for estimation
                    Options: 'FCF', 'PB', 'PE', 'PR'
    :param projection_period: (str) the time frame to project growth over ('5yr')
    :param verbose: (bool) if True, info for security used will be printed
    :param pe_growth: (str) can be 'historic' or 'estimate'
                    Default is estimate because it uses analyst estimates for future
                    growth in calculation, and not just historic growth.

    All of the variables below only need to be passed if using function repeatedly,
    as when using get_intrinsic_range, which passes these automatically. This avoids
    too many repetitive API calls. This means most users will not need to pass these 
    values into this function.

    :param beta: (float) if not passed, will be requested through API
    :param rfr: (float) risk-free rate of return, API queried if not passed
    :param market_return: (float) the market return, will be calculated if not passed
    :param shares_out: (float) the number of outstanding shares, API used if None
    :param price: (float) price of the security, will be gotten from API if not passed
    :param eps: (float) EPS value to be used, will be gotten from API if not passed
    """
    if rfr is None or market_return is None:
        rfr, market_return = get_market_info(c, projection_period=projection_period)
        if rfr is None and market_return is None:
            pass
        else:
            print("Retrieving market info, pass rfr and market_return to skip this")        

    if beta is None:
        # Get fundamentals for ticker
        print("getting beta")
        beta = get_stock_data(c, ticker, 'fundamentals').loc[ticker]['beta']
    
    capm = rfr + beta * (market_return - rfr)
    
    def compound_int(principal, interest_rate, periods, per_period=1.0):
        amount = principal * (1 + (interest_rate/per_period))**(periods*per_period)
        return amount
    
    if method == 'FCF':
        if projection_period == '5yr':
            try:
                fcf = float(fetch_metric(ticker, 'Free Cash Flow', root_dir, 'fundies_yearly'))
            except:
                print('No FCF data for {}, skipping'.format(ticker))
                return None
            fcf_growth = float(fetch_metric(ticker, 'FCF Growth 5yr', root_dir, 'fundies'))
            fcf_proj = compound_int(fcf, fcf_growth, 5.0)
            if verbose:
                print('Recent FCF:', fcf)
                print('FCF Growth 5 yr:', fcf_growth)
                print('FCF Projection:', fcf_proj)
                print('Risk-free Rate:', rfr)
                print('Beta:', beta)

            # Get industry average from sp500
            #sector = list(pd.read_excel('export-net.xlsx', skiprows=1)['Symbol'])
            sector = [ticker]
            pfcfs = []
            for company in sector:
                try:
                    fcf = float(fetch_metric(company, 'Free Cash Flow', root_dir, 'fundies_yearly'))
                except:
                    continue
                else:
                    #print(company)
                    if price is None:
                        price = c.get_quote(company).json()[company]['lastPrice']
                    pfcf = price/fcf
                    pfcfs.append(pfcf)

            sector_avg_pfcf = np.mean(pfcfs)
            #print(sector_avg_pfcf)

            estimate = fcf_proj * 0.95 * sector_avg_pfcf * 0.90 * (1 - capm)
            if verbose:
                print('Intrinsic Value', estimate)
        
    if method == 'PE':
        if projection_period == '5yr':
            try:
                valuation = pd.read_csv(root_dir+'/{}/valuation.csv'.format(ticker)).set_index('Unnamed: 0')
                try:
                    industry_avg_pe = valuation.loc['Price/Earnings (TTM, GAAP)']['Industry']
                except:
                    try:
                        industry_avg_pe = valuation.loc['Price/Earnings (TTM, GAAP)'][ticker]
                    except:
                        print('No P/E information available for {}, skipping'.format(ticker))
                        return None
                 # Filter out seemingly too large/nonsensical values sometimes found on website
                if industry_avg_pe > 100:
                    industry_avg_pe = valuation.loc['Price/Earnings (TTM, GAAP)'][ticker]
            except:
                try:
                    fundies = pd.read_csv(root_dir+'/{}/fundies.csv'.format(ticker)).set_index('Unnamed: 0')
                    industry_avg_pe = fundies.loc['Price/Earnings (TTM)']
                except:
                    print('No P/E information available for {}, skipping'.format(ticker))
                    return None

            if pe_growth == 'historic':
                try:
                    eps_growth = float(fetch_metric(ticker, 'EPS Growth 5yr', root_dir, 'fundies'))
                except:
                    print('EPS Growth 5yr not found for {}'.format(ticker))
                    return None
            if pe_growth == 'estimate':
                try:
                    eps_growth = float(fetch_metric(ticker, 'Growth 5yr Actual/Est', root_dir, 'earnings'))
                except:
                    print('Growth 5yr Actual/Est not found for {}'.format(ticker))
                    return None
            if np.isnan(eps_growth):
                print('No EPS Growth for {}, skipping'.format(ticker))
                return None
            if eps is None:
                eps = get_stock_data(c, ticker, 'fundamentals').loc[ticker]['epsTTM']
            eps_proj = compound_int(eps, eps_growth, 5.0)
            
            estimate = eps_proj * 0.95 * industry_avg_pe * 0.9 * (1 - capm)
            
            if verbose:
                print('EPS Growth 5yr:', eps_growth)
                print('EPS TTM:', eps_ttm)
                print('EPS Projection:', eps_proj)
                print('Industry Avg P/E:', industry_avg_pe)
                print('Intrinsic Value:', estimate)

    if method == 'PR':
        if projection_period == '5yr':
            
            try:
                valuation = pd.read_csv(root_dir+'/{}/valuation.csv'.format(ticker)).set_index('Unnamed: 0')
                try:
                    industry_avg_ps = valuation.loc['Price/Sales (TTM)']['Industry']
                except:
                    try:
                        industry_avg_ps = valuation.loc['Price/Sales (TTM)'][ticker]
                    except:
                        print('No Price/Sales data for {}, skipping'.format(ticker))
                        return None
                # Filter out seemingly too large/nonsensical values sometimes found on website
                if industry_avg_ps > 100:
                    industry_avg_ps = valuation.loc['Price/Sales (TTM)'][ticker]
            except:
                try:
                    fundies = pd.read_csv(root_dir+'/{}/funies.csv'.format(ticker)).set_index('Unnamed: 0')
                    industry_avg_ps = fundies.loc['Price/Sales (TTM)']
                except:
                    print('No Price/Sales data for {}, skipping'.format(ticker))
                    return None
                
            try:
                rev_growth = float(fetch_metric(ticker, 'Revenue Growth 5yr', root_dir, 'fundies'))
            except:
                print('EPS Growth 5yr not found for {}'.format(ticker))
                return None
            
            try:
                rev = float(fetch_metric(ticker, 'Total Revenue', root_dir, 'fundies_yearly'))
            except:
                rev = float(fetch_metric(ticker, 'NetInterIn after Loan Loss Provision',
                                         root_dir, 'fundies_yearly'))
            rev_proj = compound_int(rev, rev_growth, 5.0)

            
            if shares_out is None:
                shares_out = get_stock_data(c, ticker, 'fundamentals').loc[ticker]['sharesOutstanding']

            estimate = rev_proj * 0.95 * industry_avg_ps * 0.9 * (1 - capm) / shares_out

            if verbose:
                print('Rev Growth 5yr:', rev_growth)
                print('Current Revenue:', rev)
                print('Rev Projection:', rev_proj)
                print('Industry Avg P/R:', industry_avg_ps)
                print('Intrinsic Value:', estimate)

    if method == 'PB':
        if projection_period == '5yr':
            try:
                valuation = pd.read_csv(root_dir+'/{}/valuation.csv'.format(ticker)).set_index('Unnamed: 0')
                try:
                    industry_avg_pb = valuation.loc['Price/Book (MRQ)']['Industry']
                except:
                    try:
                        industry_avg_pb = valuation.loc['Price/Book (MRQ)'][ticker]
                    except:
                        print('Price/Book info not available for {}, skipping'.format(ticker))
                        return None
                 # Filter out seemingly too large/nonsensical values sometimes found on website
                if industry_avg_pb > 100:
                    industry_avg_pb = valuation.loc['Price/Book (MRQ)'][ticker]
            except:
                print('Industry average not found, using company P/B (MRQ)')
                try:
                    fundies = pd.read_csv(root_dir+'/{}/fundies.csv'.format(ticker)).set_index('Unnamed: 0')
                    industry_avg_pb = fundies.loc['Price/Book (MRQ)']
                except:
                    print('Price/Book info not available for {}, skipping'.format(ticker))
                    return None

            fundies_yrly = pd.read_csv(root_dir+'/{}/fundies_yearly.csv'.format(ticker)).set_index('Unnamed: 0')
            fundies_yrly = fundies_yrly[[col for col in fundies_yrly.columns if col not in ['Report']]]
            fundies_yrly = fundies_yrly.T
            new_df = pd.DataFrame()
            for col in ['Total Assets', 'Total Liabilities']:
                new_df[col] = fundies_yrly[col].astype('float64')
            fundies_yrly['Book Value'] = new_df['Total Assets'] - new_df['Total Liabilities']
            #print(fundies_yrly['Book Value'])
            fundies_yrly['Book Growth'] = fundies_yrly['Book Value'].pct_change()
            book_growth = fundies_yrly['Book Growth'].mean()
            assets = float(fetch_metric(ticker, 'Total Assets', root_dir, 'fundies_yearly'))
            liabilities = float(fetch_metric(ticker, 'Total Liabilities', root_dir, 'fundies_yearly'))
            book = assets - liabilities 
            book_proj = compound_int(book, book_growth, 5.0)
            if shares_out is None:
                shares_out = get_stock_data(c, ticker, 'fundamentals').loc[ticker]['sharesOutstanding']

            estimate = book_proj * 0.95 * industry_avg_pb * 0.9 * (1 - capm) / shares_out
            
            if verbose:
                print('Book Growth 5yr:', book_growth)
                print('Current Book:', book)
                print('Book Projection:', book_proj)
                print('Industry Avg P/B:', industry_avg_pb)
                print('Intrinsic Value:', estimate)

    return estimate

def compound_int(principal, interest_rate, periods, per_period=1):
        amount = principal * (1 + (interest_rate/per_period))**(periods*per_period)
        return amount
    
def get_intrinsic_range(c, tickers, root_dir, side=None, show_prices=True):
    if type(tickers) == str:
        tickers = [tickers]
    estimates = {}
    #if show_prices:
    rfr, mr = get_market_info(c, projection_period='5yr')
    stock_data = get_stock_data(c, tickers, 'both')[['lastPrice', 
                                                        'beta', 
                                                        'sharesOutstanding',
                                                        'epsTTM'
                                                       ]]
    prices = stock_data['lastPrice']
    betas = stock_data['beta']
    shares_out = stock_data['sharesOutstanding']
    eps = stock_data['epsTTM']
    for ticker in tickers:
        success = False
        tries = 0
        while not success:
            try:
                estimates[ticker] = {}

                for method in ['FCF', 'PE', 'PR', 'PB']:
                    estimate = calculate_intrinsic(c, ticker, root_dir, method, 
                                                   rfr=rfr, 
                                                   market_return=mr,
                                                   beta=betas.loc[ticker],
                                                   shares_out=shares_out.loc[ticker],
                                                   price=prices.loc[ticker],
                                                   eps=eps.loc[ticker]
                                                  )
                    estimates[ticker]['Intrinsic: '+method] = estimate
                values = pd.Series(list(estimates[ticker].values())).dropna()
                if len(values) >= 1:
                    try:
                        low = min(values)
                        high = max(values)
                    except:
                        for item in values:
                            print(item)
                        raise
                else:
                    low = np.NaN
                    high = np.NaN
                estimates[ticker]['Est Low'] = low
                estimates[ticker]['Est High'] = high
                if show_prices:
                    estimates[ticker]['Price'] = prices.loc[ticker]
                success = True
                
            except:
                raise
                print(estimates)
                print("Could not do {} on attempt {}".format(ticker, tries))
                tries += 1
                time.sleep(1)
                if tries >= 5:
                    break
                continue
        #time.sleep(.2)

    estimates = pd.DataFrame.from_dict(estimates, orient='columns')
    estimates = estimates.T
    
    if side == 'long':
        estimates['Margin of Safety'] = estimates['Est Low'] / estimates['Price'] - 1
    elif side == 'short':
        estimates['Margin of Safety'] = estimates['Price'] / estimates['Est High'] - 1

    return estimates

def clean(x, show_errors=False):
    """
    This function is used to clean strings containing numeric data of the 
    common issues found in TD Ameritrade's website
    """
    if isinstance(x, str):
        check = re.split('/|-|, ', x)
        x = x.strip()
        x = x.replace(',','')
        if x == '--':
            x = np.NaN
        elif x.startswith('(') and x.endswith(')'):
            x = x.strip('(').strip(')')
            if x.endswith('%'):
                x = np.float(x.strip('%'))/100
            if x.startswith('$'):
                x = x.strip('$')
            x = -np.float(x)
        elif x.endswith('%'):
            x = np.float(x.strip('%'))/100
        elif x.endswith('x'):
            x = x.strip('x')
            if x == '--':
                x = np.NaN
            else:
                x = np.float(x)
        elif x.startswith('$') or x.startswith('-$'):
            x = np.float(x.replace('$',''))
        elif len(check) > 1 and check[-1].isdigit():
            if x.startswith('(Unconfirmed)'):
                x = x.replace('(Unconfirmed) ','')
            x = pd.to_datetime(x, infer_datetime_format=True)
        else:
            try:
                x = float(x)
            except:
                if show_errors:
                    print(x) 
                x = np.NaN
    return x            

def fetch_metric(ticker, metric, root_dir, file_name, year=None):
    if year is None:
        year = datetime.now().year
    
    if file_name == 'fundies':
        df = retrieve_df(ticker, root_dir, file_name)
        try:
            datum = df.loc[metric, ticker]
        except:
            raise
    
    elif file_name == 'fundies_yearly':
        df = retrieve_df(ticker, root_dir, file_name)
        if pd.notnull(df.loc[metric, str(year)]):
            datum = df.loc[metric, str(year)]
        else:
            try:
                datum = df.loc[metric, (str(year-1))]
            except:
                raise
    
    elif file_name == 'earnings':
        df = retrieve_df(ticker, root_dir, file_name)
        try:
            datum = df.loc[metric, ticker]
        except:
            raise
    
    return datum
    
def get_market_info(c, projection_period='5yr'):
    if projection_period == '5yr':
        # Get risk-free rate:
        rfr = c.get_quote('$FVX.X').json()['$FVX.X']['lastPrice'] / 1000

        # Get 5 yr avg yearly market growth
        r = c.get_price_history(symbol='$SPX.X',
                                period=client.Client.PriceHistory.Period.FIFTEEN_YEARS,
                                period_type=client.Client.PriceHistory.PeriodType.YEAR,
                               ).json()
        df = pd.DataFrame(r['candles'])
        df = df.iloc[-61:] # set to an extra month back to leave out most recent month
        # Make new dataframe to copy yearly rows to
        df2 = pd.DataFrame()
        for i in range(len(df)):
            if i % 12 == 0:
                df2 = pd.concat([df2, df.iloc[i]], axis=1, sort=True)
        df2 = df2.T
        df2['growth'] = df2.close.pct_change()
        market_return = df2.growth.mean()
        
        return rfr, market_return

def get_stock_data(c, instruments, query='both', step_size=250, verbose=False):
    '''
    This function uses the TD Ameritrade API to get data
    
    :param c: tda-api client
    :param instruments: (list-like or DataFrame) list of ticker symbols.
                        Can also pass a DataFrame with index of symbols
    :param query: (string) 'fundamentals', 'quotes', or 'both'
    :param step_size: (int) how many securities to query at once
    '''
    # Adjust for a single string
    if type(instruments) == str:
        instruments = [instruments]
    total_num = len(instruments)
    # Allows returned stock data to be concatenated to an input dataframe
    if type(instruments) == pd.DataFrame:
        data = instruments.copy()
        instruments = list(sorted(instruments.index))
    # Create new dataframe if instruments is a list
    else:
        data = pd.DataFrame()
        instruments = sorted(instruments)
    if verbose:
        print("Getting {} for {} instruments".format(query, total_num))
    low = 0
    
    # Helper functions to be used in loop:
    def get_fundies(ticks, errs):
        if len(ticks) == 1 and type(ticks) == list:
            ticks = [ticks[0]]
        if errs > 10:
            print("More than 10 errors on tickers {}".format(ticks))
            return None
        r = c.search_instruments(ticks,
                                 projection=client.Client.Instrument.Projection.FUNDAMENTAL)
        if not r.ok:
            print("Fundamentals problem for {}".format(ticks))
            errs += 1
            time.sleep(3)
            get_fundies(ticks, errs)
        return r
    def get_quotes(ticks, errs):
        if len(ticks) == 1 and type(ticks) == list:
            ticks = [ticks[0]]
        if errs > 10:
            print("More than 10 errors on tickers {}".format(ticks))
            return None
        r = c.get_quotes(ticks)
        if not r.ok:
            print("Quote problem for {}".format(ticks))
            errs += 1
            time.sleep(3)
            get_quotes(ticks, errs)
        return r

    # Run a query for a number of tickers equal to step_size
    for i in range(total_num//step_size + 1):
        high = low + step_size
        if high > total_num:
            high = total_num
        ticks = list(instruments[low:high])
        errs = 0
        
        if query == 'fundamentals' or query == 'both':
            errs = 0
            # Get response from API
            r = get_fundies(ticks, errs)
            # Convert to dataframe
            temp = pd.DataFrame.from_dict([r.json()[x]['fundamental'] for x in r.json().keys()], orient='columns')
            # drop peRatio in favor of more current value from quote query
            if query == 'both':
                temp.drop(columns=['peRatio'], inplace=True)
            temp['Symbol'] = r.json().keys()
            # drop duplicate symbol column and set index
            temp.drop(columns='symbol', inplace=True)
            temp.set_index('Symbol', inplace=True)

        if query == 'quotes' or query == 'both':
            errs = 0
            # Get response from API
            r = get_quotes(ticks, errs)
            # Convert to dataframe
            temp2 = pd.DataFrame.from_dict([r.json()[x] for x in r.json().keys()], orient='columns')
            temp2['Symbol'] = r.json().keys()
            # Drop duplicate symbol column and set index
            temp2.drop(columns='symbol', inplace=True)
            temp2.set_index('Symbol', inplace=True)

        # Concat to dataframe
        if query == 'fundamentals':
            data = pd.concat([data, temp], axis=0)
        elif query == 'quotes':
            data = pd.concat([data, temp2], axis=0)
        elif query == 'both':
            temp = pd.concat([temp, temp2], axis=1, join='inner', sort=True)
            data = pd.concat([data, temp], axis=0)

        # Increment variable for next step
        low+=step_size
    
    return data

def get_keys(path):
    """
    This function will get dictionary of keys from a stored json file

    :param path: (str) directory path for the .json file with keys
    """
    with open(path) as f:
        return json.load(f)

def get_tab_links(driver):
    """
    (Depracated) Function used to get html links for each tab on a stock's
    page. Replaced in favor of using xpaths with selenium, as pressing the
    buttons does not reload the entire webpage, as is done when using a
    html link.

    :param driver: (Selenium webdriver) driver returned from start_bot()
    """
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    stockTabs = soup.find_all('nav', {'class': 'stockTabs'})
    info = stockTabs[0].find_all('a')
    texts = [x.get_text() for x in info]
    links = [x.get('href') for x in info]
    tabs = dict(zip(texts,links))
    return tabs

def get_tickers(index):
    '''
    Function to generate a dataframe of securities traded based on exchange.
    Uses data from old.nasdaq.com

    :param index: What group of tickers to gather:
                    Options: 'sp500', nsdq', 'nyse', 'amex', 'all'
    '''
    def filter_stocks(dataframe):
        """
        This function formats the dataframe produced by get_tickers() to function with the
        tdameritrade API
        """
        dataframe.Symbol = dataframe.Symbol.astype('str')
        dataframe.Symbol = dataframe.Symbol.map(lambda x: x.strip())
        dataframe = dataframe[dataframe.Symbol != 'ACCP']
        print("Number of tickers before trim:", len(dataframe))
        dataframe = dataframe[dataframe.Symbol.str.isalpha()]
        print("Number of tickers after trim:", len(dataframe))
        dataframe = dataframe.set_index('Symbol')
        return dataframe

    if index not in ['nsdq', 'nyse', 'amex', 'sp500', 'all']:
        raise ValueError("Unrecognized index given.")
    elif index == 'nsdq':
        nsdq = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')
        return filter_stocks(nsdq)
    elif index == 'nyse':
        nyse = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download')
        return filter_stocks(nyse)
    elif index == 'amex':
        amex = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download')
        return filter_stocks(amex)
    elif index == 'sp500':
        sp500 = pd.read_csv('sp500.txt')
        return filter_stocks(sp500)
    elif index == 'all':
        nsdq = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')
        nyse = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download')
        amex = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download')
        nsdq['Exchange'] = 'NASDAQ'
        nyse['Exchange'] = 'NYSE'
        amex['Exchange'] = 'AMEX'
        us_stocks = pd.concat([nsdq, nyse, amex], axis=0)
        # Removing duplicate entries:
        us_stocks.drop_duplicates(subset='Symbol',inplace=True)
        return filter_stocks(us_stocks)

def reduce_tabs(driver):
    """
    This function is used when an action opens the result on a new tab, in
    order to reduce the number of browser tabs back to 1, and switch to the
    intended tab.

    :param driver: (Selenium webdriver)
    """
    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[0])
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

def retrieve_df(ticker, root_dir, names):
    dfs = []
    if type(names) == str:
        names = [names]
    for name in names:
        path = root_dir + '/{}/{}.csv'.format(ticker, name)
        df = pd.read_csv(path).set_index('Unnamed: 0')
        dfs.append(df)
    
    if len(dfs) == 1:
        dfs = dfs[0]
    
    return dfs

def search_symbol(driver, ticker):
    """
    This function searches for a ticker symbol on TD Ameritrade website once
    user is logged in.

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to search
    """

    # Attempt the more expedient symbol lookup, rever to main search otherwise
    try:
        search = driver.find_element_by_xpath('//*[@id="symbol-lookup"]')
        search.click()
        search.clear()
    except:
        driver.switch_to.default_content()
        search = driver.find_element_by_name("search")
        kind = 'search'
    else:
        kind = 'symbol'
    # Enter ticker symbol to search and click search button
    search.send_keys(ticker)
    if kind == 'symbol':
        driver.find_element_by_xpath('//*[@id="layout-full"]/div[1]/div/div[1]/div/a').click()
    elif kind == 'search':
        driver.find_element_by_id("searchIcon").click()
    # Give extra time for webpage to load
    time.sleep(4)

def scrape_analysts(driver, ticker, search_first=True):
    """
    This function scrapes the "Analyst Reports" tab of a TD Ameritrade security
    lookup page

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    :param search_first: (bool) allows for chain of scrapes to be done on one
                                security when set to False. Leave set to True
                                unless you are sure you are already on the
                                desired security, or the wrong data will scrape
    """
    # Search symbol first if flag is True
    if search_first:
        search_symbol(driver, ticker)

    # Find iframe with tabs (main iframe)
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])

    # Switch to Analyst Reports tab
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Summary'))
    driver.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[8]/a').click()
    time.sleep(1)

    # Wait for conditions before soup is made
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Archived Reports'))

    # Make soup and find container and elements
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    contain = soup.find('table', {'class':'ui-table provider-table'}).find('tbody')
    trs = contain.find_all('tr')

    analysts = []
    ratings = []
    dates = []
    for tr in trs:
        divs = tr.find_all('div')
        analyst = divs[0].get('class')[1].strip()
        
        try:
            # Skip vickers
            if analyst == 'vickers':
                continue
            # Special treatment for marketEdge
            else:
                # Get date or NaN otherwise
                try:
                    txt = tr.find('p', {'class':'rating-since'}).get_text()
                    date = txt.replace('Rating Since ','')
                except:
                    date = NaN
                # Special treatment for marketEdge
                if analyst == 'marketEdge':
                    analysts.append(analyst+' opinion')
                    rating = divs[2].get('class')[2]
                    ratings.append(rating)
                    dates.append(date)
                    flag = False
                    i = 0
                    while flag == False:
                        i += 1
                        rating = divs[3].get('class')[1][-i].strip()
                        try:
                            rating = float(rating)
                            if i != 1:
                                rating = -rating
                            flag = True
                        except:
                            flag = False
                # Special treatment for cfra
                elif analyst == 'cfra':
                    rating = divs[2].get('class')[1][-1].strip()
                    try:
                        int(rating)
                    except:
                        rating = np.NaN
                else:
                    rating = divs[2].get('class')[1].strip()
                # Try to make ratings numeric
                try:
                    rating = int(rating)
                except:
                    rating = rating
        except:
            rating = np.NaN
            date = np.NaN

        analysts.append(analyst)
        ratings.append(rating)
        dates.append(date)

    # Create dataframe
    analyst_dict = dict(zip(analysts,zip(ratings,dates)))
    temp = pd.DataFrame.from_dict(analyst_dict, 
                                  orient='index', 
                                  columns=[ticker,'Rating Since'],
                                  )
    # Convert date column to datetime
    temp['Rating Since'] = pd.to_datetime(temp['Rating Since'], infer_datetime_format=True)
    
    return temp
    
def scrape_earnings(driver, ticker, search_first=True):
    """
    This function scrapes the "Earnings" tab of a TD Ameritrade security
    lookup page

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    :param search_first: (bool) allows for chain of scrapes to be done on one
                                security when set to False. Leave set to True
                                unless you are sure you are already on the
                                desired security, or the wrong data will scrape
    """
    # Search for symbol if flag is True
    if search_first:
        search_symbol(driver, ticker)
    
    # Find main iframe:  
    driver.switch_to.default_content()    
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])
    
    # Switch to Earnings tab:
    WebDriverWait(driver,10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[4]/a')).click()
    time.sleep(1)
    
    # Switch to Earnings Analysis (1st sub tab)
    WebDriverWait(driver,10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/div[4]/nav/nav/a[1]')).click()
    time.sleep(1)
    
    # Wait for conditions before making soup
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Annual Earnings History and Estimates'))
    element = driver.find_element_by_xpath('//*[@id="main-chart-wrapper"]')
    WebDriverWait(driver, 10).until(lambda x: EC.visibility_of_element_located(element))
    
    # Make soup and find container/elements
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    earn_dict = {}
    earnings_dict = {}
    contain = soup.find('div', {'data-module-name':'EarningsAnalysisModule'})
    header = contain.find('div', {'class':'row contain earnings-data'})
    #key = header.find('td', {'class':'label bordered'}).get_text()
    earn_dict['Next Earnings Announcement'] = header.find('td', {'class':'value week-of'}).get_text()
    
    # Get number of analysts reporting on security
    analysts = header.find_all('td', {'class':'label'})[1].get_text().split()
    for word in analysts:
        # The number of analysts will be the only numerical string
        try:
            earn_dict['Growth Analysts'] = float(word)
        except:
            continue
    # Find chart object in container, then bars
    chart = contain.find('div', {'id':'main-chart-wrapper'})
    bars = chart.find_all('div', {'class':'ui-tooltip'})
    for bar in bars:
        text = bar.get_text('|').split('|')
        # text[0] is the year
        year = text[0]
        earnings_dict[year] = {}
        # There is more text when there is a earnings surprise
        if len(text) > 4:
            earnings_dict[year]['Earnings Result'] = text[1]
            earnings_dict[year][text[2].strip('"').strip().strip(':')] = float(text[3].replace('$',''))
            earnings_dict[year][text[4].split(':')[0]] = text[4].split(':')[1].strip()
        else:
            earnings_dict[year]['Earnings Result'] = 'Neutral'
            # Should be a string: 'Actual' or 'Estimate'
            est_string = text[1].strip('"').strip().strip(':')
            # The actual consensus estimate
            est = float(text[2].replace('$',''))
            earnings_dict[year][est_string] = est
            # Should be a string: 'Estimate range'
            est_range_string = text[3].split(':')[0]
            # The estimate range as a string
            est_range = text[3].split(':')[1].strip()
            # Convert to 
            earnings_dict[year][est_range_string] = est_range
            
    # Create df and all useful columns
    earnings_yrly = pd.DataFrame.from_dict(earnings_dict, orient='index')
    earnings_yrly['Growth'] = earnings_yrly['Actual'].pct_change()
    earnings_yrly['Low Estimate'] = earnings_yrly['Estimate range'].map(lambda x: float(x.split()[0].replace('$','')), na_action='ignore')
    earnings_yrly['Low Growth Est'] = earnings_yrly['Low Estimate'].pct_change()
    earnings_yrly['High Estimate'] = earnings_yrly['Estimate range'].map(lambda x: float(x.split()[2].replace('$','')), na_action='ignore')
    earnings_yrly['High Growth Est'] = earnings_yrly['High Estimate'].pct_change()
    # Take average of high and low for years where 'Estimate' not available
    earnings_yrly['Consensus Estimate'] = (earnings_yrly['High Estimate'] + earnings_yrly['Low Estimate']) / 2
    # Supercede these values where consensus estimates are available
    idx_to_change = earnings_yrly[earnings_yrly['Estimate'].notnull()].index
    earnings_yrly.loc[idx_to_change, 'Consensus Estimate'] = earnings_yrly.loc[idx_to_change, 'Estimate']
    # Make new column that contains the actuals and consensus estimates
    earnings_yrly['Actual/Estimate'] = earnings_yrly['Actual']
    earnings_yrly.loc[idx_to_change, 'Actual/Estimate'] = earnings_yrly.loc[idx_to_change, 'Estimate']
    earnings_yrly['A/E Growth'] = earnings_yrly['Actual/Estimate'].pct_change()

    if 'Consensus estimate' in earnings_yrly.columns:
        # Sometimes ranges aren't given, and Consensus estimate given instead, fill holes caused
        earnings_yrly['Consensus Estimate'].fillna(earnings_yrly[earnings_yrly['Consensus estimate'].notnull()]['Consensus estimate'].map(lambda x: float(x.replace('$',''))), inplace=True)
        earnings_yrly.drop(columns=['Consensus estimate'], inplace=True)
    earnings_yrly.drop(columns=['Estimate range'], inplace=True)
    earnings_yrly['Consensus Growth Est'] = (earnings_yrly['High Growth Est']+earnings_yrly['Low Growth Est']) / 2
    
    low_1yr_growth_est = earnings_yrly.iloc[-2,:]['Low Growth Est']
    high_1yr_growth_est = earnings_yrly.iloc[-2,:]['High Growth Est']
    cons_1yr_growth_est = earnings_yrly.iloc[-2,:]['Consensus Growth Est']
    growth_2yr_low_est = earnings_yrly.iloc[-2:]['Low Growth Est'].mean()
    growth_2yr_high_est = earnings_yrly.iloc[-2:]['High Growth Est'].mean()
    growth_2yr_cons_est = (growth_2yr_low_est + growth_2yr_high_est) / 2
    earn_dict['Growth 1yr Low Est'] = low_1yr_growth_est
    earn_dict['Growth 1yr High Est'] = high_1yr_growth_est
    earn_dict['Growth 1yr Consensus Est'] = cons_1yr_growth_est
    earn_dict['Growth 2yr Low Est'] = growth_2yr_low_est
    earn_dict['Growth 2yr High Est'] = growth_2yr_high_est
    earn_dict['Growth 2yr Consensus Est'] = growth_2yr_cons_est
    earn_dict['Growth 5yr Low Est'] = earnings_yrly['Low Growth Est'].mean()
    earn_dict['Growth 5yr High Est'] = earnings_yrly['High Growth Est'].mean()
    earn_dict['Growth 5yr Consensus Est'] = earnings_yrly['Consensus Growth Est'].mean()
    earn_dict['Growth 5yr Actual/Est'] = earnings_yrly['A/E Growth'].mean()
    earn_dict['Growth 3yr Historic'] = earnings_yrly['Growth'].mean()

    earn_df = pd.DataFrame.from_dict(earn_dict, orient='index', columns=[ticker])
    earn_df[ticker] = earn_df[ticker].map(clean)

    return earn_df, earnings_yrly
        
def scrape_fundamentals(driver, ticker, search_first=True):
    """
    This function scrapes the "Fundamentals" tab of a TD Ameritrade security
    lookup page

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    :param search_first: (bool) allows for chain of scrapes to be done on one
                                security when set to False. Leave set to True
                                unless you are sure you are already on the
                                desired security, or the wrong data will scrape
    """
    # Search symbol first if flag is True
    if search_first:
        search_symbol(driver, ticker)
        #tabs = get_tab_links()
    
    # Gets Overview
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])
    WebDriverWait(driver,10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[5]/a')).click()
    #time.sleep(1)
    WebDriverWait(driver,10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/div[4]/nav/nav/a[1]')).click()
    time.sleep(1)
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])
    #WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Price Performance'))
    #driver.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[5]/a').click()
    #time.sleep(1)

    # Wait for conditions before making soup
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Price Performance'))
    element = driver.find_element_by_xpath('//*[@id="price-charts-wrapper"]/div')
    WebDriverWait(driver, 10).until(lambda x: EC.visibility_of_element_located(element))
      
    # Make soup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Scrapes current valuation ratios
    contain = soup.find('div', {'class': 'ui-description-list'})
    labels = [x.get_text().strip() for x in contain.find_all('dt')]
    values = [float(x.get('data-rawvalue')) for x in contain.find_all('dd')]
    for i, value in enumerate(values):
        if value == '-99999.99' or value == -99999.99:
            values[i] = np.NaN
    fundies = dict(zip(labels,values))

    # Gets 5yr low and high from chart
    contain = soup.find('div', {'class':'col-xs-8 price-history-chart'})
    five_yr = contain.find_all('div', {'class':'marker hideOnHover'})
    fundies['5yr Low'] = five_yr[0].get_text().split(' ')[1]
    fundies['5yr High'] = five_yr[1].get_text().split(' ')[1]

    # Gets 5 year Price Performance data from each hover section of graphic
    periods = contain.find_all('div', {'class':'period'})
    texts = [x.get_text('|') for x in periods]
    past_dict = {}
    yr_growths = []
    for text in texts:
        parts = text.split('|')
        year = parts[2].split(' ')[3].strip()
        past_dict[year] = {}
        high = parts[1].split(' ')[2].strip()
        low = parts[0].split(' ')[2].strip()
        change = parts[2].split(' ')[0].strip()
        past_dict[year]['high'] = high
        past_dict[year]['low'] = low
        past_dict[year]['change'] = change
        yr_growths.append(float(change.strip('%')))
    fundies['5yr Avg Return'] = np.mean(yr_growths) / 100

    # Gets Historic Growth and Share Detail
    contain = soup.find('div', {'data-module-name':'HistoricGrowthAndShareDetailModule'})
    boxes = contain.find_all('div', {'class':'col-xs-4'})
    labels = []
    values = []
    historic_data = True

    for box in boxes:
        numbers = []
        words = []
        if box.find('h4').get_text() == 'Historic Growth':
            for dt in box.find_all('dt')[1:]:
                word = dt.get_text('|').split('|')[0].strip() +' Growth 5yr'
                words.append(word)
            for dd in box.find_all('dd'):
                try:
                    number = float(dd.find('label').get('data-value'))
                    #print(number)
                    if number == -99999.99:
                        #print("here")
                        number = np.NaN
                except:
                    #print("didn't find number")
                    try:
                        number = dd.find('span').get_text()
                    except:
                        number = np.NaN
                #print(number)
                numbers.append(number)
            if len(words) == 0:
                print("Historic Growth not available for {}".format(ticker))
                historic_data = False
        else:
            for dt in box.find_all('dt')[1:]:
                word = dt.get_text('|').split('|')[0].strip()
                words.append(word)
            for dd in box.find_all('dd'):
                try:
                    number = float(dd.get('data-rawvalue'))
                    if number == -99999.99:
                        number = np.NaN
                except:
                    try:
                        number = dd.get_text()
                    except:
                        number = np.NaN
                numbers.append(number)

        labels = labels + words
        values = values + numbers
    
    # Make df of Historic Growth and Share Detail
    fundies2 = dict(zip(labels, values))

    # Get ready to scrape financial reports:
    report_names = ['Balance Sheet',
              'Income Statement',
              'Cash Flow'
             ]
    xpaths = [#'//*[@id="layout-full"]/div[4]/nav/nav/a[1]', # Already done
              '//*[@id="layout-full"]/div[4]/nav/nav/a[2]',
              '//*[@id="layout-full"]/div[4]/nav/nav/a[3]',
              '//*[@id="layout-full"]/div[4]/nav/nav/a[4]'
             ]
    reports = dict(zip(report_names, xpaths))
    
    # Function to scrape each report, since their formats are similar enough
    def scrape_report(name, xpath):
        # Switch to Appropriate Report
        driver.find_element_by_xpath(xpath).click()
        time.sleep(1)
        iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
        driver.switch_to.frame(iframes[3])
        driver.switch_to.default_content()
        iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
        driver.switch_to.frame(iframes[3])
        WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Values displayed are in millions.'))
        element = driver.find_element_by_xpath('//*[@id="layout-full"]/div[4]/div/div')
        WebDriverWait(driver, 10).until(lambda x: EC.visibility_of_element_located(element))

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        #pprint.pprint(soup)
        contain = soup.find('div', {'data-module-name':'FinancialStatementModule'})
        year_info = [x.get_text('|') for x in contain.find_all('th', {'scope':'col'})]
        years = [x.split('|')[0] for x in year_info]
        dates = [x.split('|')[1] for x in year_info]

        sheet = {}
        for i, year in enumerate(years):
            sheet[year] = {}
            sheet[year]['Date'] = dates[i]
        row_names = []
        contain = soup.find('div', {'class':'row contain data-view'})
        rows = contain.find_all('tr')[1:] # Skips the header row
        #rows = contain.find_all('th', {'scope':'row'})
        for row in rows:
            #print(row)
            row_name = row.get_text('|').split('|')[0]
            row_names.append(row_name)
            values = row.find_all('td')
            for i, value in enumerate(values):
                sheet[years[i]][row_name] = value.get_text()
                
        temp = pd.DataFrame.from_dict(sheet, orient='index').T
        temp['Report'] = name
        return temp
    
    
    # Create summary dataframes
    temp = pd.DataFrame.from_dict(fundies, orient='index', columns=[ticker])
    temp2 = pd.DataFrame.from_dict(fundies2, orient='index', columns=[ticker])
    temp2.rename(index={'Current Month':'Short Int Current Month',
                         'Previous Month':'Short Int Prev Month',
                         'Percent of Float':'Short Int Pct of Float'
                        },
                inplace=True)
    # Clean these rows if they exist
    try:
        temp2.loc['Short Int Pct of Float',:] = temp2.loc['Short Int Pct of Float',:].astype('float64') / 100
        temp2.loc['% Held by Institutions',:] = temp2.loc['% Held by Institutions',:].astype('float64') / 100
    except:
        print("Short Interest info not available for {}".format(ticker))

    # Create yearly dataframe
    yearly = pd.DataFrame.from_dict(past_dict, orient='index').T
    for name, xpath in reports.items():
        tempy = scrape_report(name, xpath)
        yearly = pd.concat([yearly, tempy], axis=0, sort=False)
    
    # Combine two summary dataframes
    temp = pd.concat([temp,temp2], axis=0) 

    # Clean data in the dataframes  
    for col in temp:
        temp[col] = temp[col].map(lambda x: clean(x),  na_action='ignore')
    colnames = [col for col in yearly.columns if col not in ['Report']]
    for col in colnames:
        yearly[col] = yearly[col].map(lambda x: clean(x), na_action='ignore')
    
    # Create FCF and growth features for summary from yearly:
    yearly = yearly.T.astype('float64', errors='ignore')
    temp = temp.T.astype('float64', errors='ignore')
    indices = [indx for indx in yearly.index if indx not in ['Report']]
    yearly['Free Cash Flow'] = np.NaN
    yearly['FCF Growth'] = np.NaN
    # Allows this to not throw errors if values not available
    try:
        yearly.loc[indices,'Free Cash Flow'] = yearly.loc[indices,'Total Cash from Operations'] + yearly['Capital Expenditures']
        yearly.loc[indices,'FCF Growth'] = yearly.loc[indices,'Free Cash Flow'].pct_change()
        temp['FCF Growth 5yr'] = yearly['FCF Growth'].mean()
    except:
        temp['FCF Growth 5yr'] = np.NaN
    # These percentages must be formatted
    if historic_data:
        temp['EPS Growth 5yr'] = temp['EPS Growth 5yr']/100
        temp['Revenue Growth 5yr'] = temp['Revenue Growth 5yr']/100
        temp['Dividend Growth 5yr'] = temp['Dividend Growth 5yr']/100
    else:
        temp['EPS Growth 5yr'] = np.NaN
        temp['Revenue Growth 5yr'] = np.NaN
        temp['Dividend Growth 5yr'] = np.NaN

    # Transposing dataframes back
    temp = temp.T
    yearly = yearly.T

    return temp, yearly

def scrape_summary(driver, ticker, search_first=True, return_full=False):
    """
    This function scrapes the "Summary" tab of a TD Ameritrade security
    lookup page

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    :param search_first: (bool) allows for chain of scrapes to be done on one
                                security when set to False. Leave set to True
                                unless you are sure you are already on the
                                desired security, or the wrong data will scrape
    :param return_full: (bool) will return dataframe with extra column containing
                               feature descriptions for the rows.
    """
    # Search symbol first if flag is True:
    if search_first:
        search_symbol(driver, ticker)
    #tabs = get_tab_links()
    #driver.get(tabs['Summary'])

    # Find main iframe
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])

    # Switch to Summary tab
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, 'Summary'))
    WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[1]/a')).click()
    
    # Wait for conditions to be met before making soup
    element = driver.find_element_by_xpath('//*[@id="stock-summarymodule"]/div/div/div[2]/div')
    WebDriverWait(driver, 10).until(lambda x: EC.visibility_of_element_located(element))
    # Add extra time for data to load
    time.sleep(1)
    
    # Make soup and find elements
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    dts = soup.find_all('dt')

    # Set flag which will be made false if no dividend is given
    dividend_given = True
    texts = []
    for dt in dts:
        try:
            texts.append(dt.get_text('|'))        
        except:
            print("error")
            continue

    dds = soup.find_all('dd')
    values = []
    for dd in dds:
        try:
            values.append(dd.get_text('|'))        
        except:
            print("error")
            continue

    fields = [x.split('|')[0] for x in texts]
    alt_info = [x.split('|')[1:] for x in texts]

    # Make dataframe and fix row names
    data_dict = dict(zip(fields,zip(alt_info,values)))
    temp = pd.DataFrame.from_dict(data_dict, orient='index')
    temp.loc['Volume', 1] = temp.loc['Volume', 0][0].strip()
    temp.rename(index={'Volume:':'Volume 10-day Avg',
                          'Volume':'Volume Past Day',
                          '10-day average volume:':'Volume',
                          'Score:':'New Constructs Score'
                        }, inplace=True)
    temp.loc['52-Wk Range', 1] = temp.loc['52-Wk Range', 0]
    price_feat = 'Closing Price'
    if price_feat not in temp.index:
        if 'Price' in temp.index:
            price_feat = 'Price'

    # Cleaning data
    if temp.loc["B/A Size",1] == '--': 
        temp = temp.append(pd.Series([[],
                                  np.NaN
                                 ],
                                 name="Bid Size"),
                      )
        temp = temp.append(pd.Series([[],
                                  np.NaN
                                 ],
                                 name="Ask Size"),
                      )
        temp = temp.append(pd.Series([[],
                                  np.NaN
                                 ],
                                 name="B/A Ratio"),
                      )
    else:
        temp = temp.append(pd.Series([[],
                                  float(temp.loc['B/A Size',1].split('x')[0])
                                 ],
                                 name="Bid Size"),
                      )
        temp = temp.append(pd.Series([[],
                                  float(temp.loc['B/A Size',1].split('x')[1])
                                 ],
                                 name="Ask Size"),
                      )
        temp = temp.append(pd.Series([[],
                                  float(temp.loc['B/A Size',1].split('x')[0])
                                            /float(temp.loc['B/A Size',1].split('x')[1])
                                 ],
                                 name="B/A Ratio"),
                    )  
    if temp.loc["Day's Range",1] == '--':
        temp = temp.append(pd.Series([[],
                                  np.NaN,
                                 ],
                                 name="Day Change $"
                                ),
                      )
        temp = temp.append(pd.Series([[],
                                  np.NaN
                                 ],
                                 name="Day Change %"
                                ),
                      )
        temp = temp.append(pd.Series([[],
                                 np.NaN
                                 ],
                                name="Day Low"),
                      )
        temp = temp.append(pd.Series([[],
                                 np.NaN
                                 ],
                                name="Day High"),
                      )
    else:
        temp = temp.append(pd.Series([[],float(temp.loc["Day's Change",1].split('|')[0].strip('|'))],
                                 name="Day Change $"
                                ),
                      )
        temp = temp.append(pd.Series([[],
                                  float(temp.loc["Day's Change",1].split('|')[2].strip('%)').strip('()'))/100
                                 ],
                                 name="Day Change %"
                                ),
                      )
        temp = temp.append(pd.Series([[],
                                 float(temp.loc["Day's Range",1].split('-')[0].strip('|').replace(',',''))
                                 ],
                                name="Day Low"),
                      )
        temp = temp.append(pd.Series([[],
                                 float(temp.loc["Day's Range",1].split('-')[1].strip('|').replace(',',''))
                                 ],
                                name="Day High"),
                      )
    if temp.loc["Annual Dividend/Yield",1] != 'No dividend':
        temp = temp.append(pd.Series([[],
                                 float(temp.loc["Annual Dividend/Yield",1].split('/')[0].strip('$'))
                                 ],
                                name="Annual Dividend $"))

        temp = temp.append(pd.Series([[],
                                 float(temp.loc["Annual Dividend/Yield",1].split('/')[1].strip('%'))/100
                                 ],
                                name="Annual Dividend %"))
    else:
        dividend_given = False
        temp = temp.append(pd.Series([[],
                                 np.NaN
                                 ],
                                name="Annual Dividend $"))
        temp = temp.append(pd.Series([[],
                                 np.NaN
                                 ],
                                name="Annual Diviend %"))
    temp.rename(columns={1:ticker}, inplace = True)
    drop = ["Day's Change", 
            "Day's Range",
            "Day's High",
            "Day's Low",
            "Avg Vol (10-day)", 
            #"52-Wk Range", 
            "Annual Dividend/Yield",
            "New Constructs Score"
            ]

    # Drop feature description column if flag is False (default)
    if return_full == False:
        temp.drop(index=drop, columns=[0], inplace=True, errors='ignore')
    
    # Clean data
    temp = temp.T
    # Only one of these columns will be present:
    try:
        temp['% Below High'] = temp['% Below High'].map(lambda x: float(x.strip('%'))/100, na_action='ignore')
    except:
        temp['% Above Low'] = temp['% Above Low'].map(lambda x: clean(x), na_action='ignore')
    
    temp['% Held by Institutions'] = temp['% Held by Institutions'].map(lambda x: clean(x)/100, na_action='ignore')
    temp['Short Interest'] = temp['Short Interest'].map(lambda x: clean(x)/100, na_action='ignore')
    # Set list of columns for cleaing
    try_to_clean = ['Prev Close',
                    'Ask close',
                    'Bid close',
                    'Beta',
                    'Ask',
                    'Bid',
                    'EPS (TTM, GAAP)',
                    'Last Trade',
                    'Last (size)',
                    price_feat,
                    'Historical Volatility',
                    'P/E Ratio (TTM, GAAP)',
                    "Today's Open",
                    'Volume',
                    'Volume 10-day Avg']
    # Clean columns
    for col in try_to_clean:
        try:
            temp[col] = temp[col].map(lambda x: clean(x), na_action='ignore')
        except:
            pass
    
    # Convert date info to datetime if it exists
    if dividend_given:
        temp['Ex-dividend'] = pd.to_datetime(temp['Ex-dividend'], infer_datetime_format=True)
    # Try to force any remaining numbers to floats:
    temp = temp.astype('float64', errors='ignore')
    temp = temp.T   
    temp.sort_index(inplace=True)

    return temp

def scrape_ticker(driver, ticker):
    """
    This function scrapes every tab of a security based on ticker passed.
    Each scrape will be attempted 5 times before being skipped, as it is 
    unlikely for the data to fail to scrape this many times unless it is 
    truly absent.

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    """
    # Getting Summary
    success = False
    tries = 0
    while not success:
        tries += 1
        try:
            summary = scrape_summary(driver, ticker)
            success = True
        except:
            print("Failed to gather summary for {} on attempt {}".format(ticker, tries))
        if tries >= 5:    
            print("Too many failed attempts for summary of {}, skipping to next df.".format(ticker))
            summary = pd.DataFrame(column=[ticker])
            break

    # Getting Earnings
    success = False
    tries = 0
    while not success:
        tries += 1
        try:
            earnings, earnings_yearly = scrape_earnings(driver, ticker, search_first=False)
            success = True
        except:
            print("Failed to gather earnings for {} on attempt {}".format(ticker, tries))
        if tries >= 5:
            print("Too many failed attempts for earnings of {}, skipping to next df.".format(ticker))
            earnings = pd.DataFrame(columns=[ticker])
            earnings_yearly = pd.DataFrame(columns=[ticker])
            break
    
    # Getting fundamentals
    success = False
    tries = 0
    while not success:
        tries += 1
        try:
            fundies, fundies_yearly = scrape_fundamentals(driver, ticker, search_first=False)
            success = True
        except:
            print("Failed to gather fundamentals for {} on attempt {}".format(ticker, tries))
        if tries >= 5:
            print("Too many failed attempts for fundamentals of {}, skipping to next df.".format(ticker))
            fundies = pd.DataFrame(columns=[ticker])
            fundies_yearly = pd.DataFrame(columns=[ticker])
            break

    # Getting valuation
    success = False
    tries = 0
    while not success:
        tries += 1
        try:
            valuation = scrape_valuation(driver, ticker, search_first=False)
            success = True
        except:
            print("Failed to gather valuation for {} on attempt {}".format(ticker, tries))
        if tries >= 5:
            print("Too many failed attempts for valuation of {}, skipping to next df.".format(ticker))
            valuation = pd.DataFrame(columns=[ticker])
            break
    
    # Getting analyst reports
    success = False
    tries = 0
    while not success:
        tries += 1
        try:
            analysis = scrape_analysts(driver, ticker, search_first=False)
            success = True
        except:
            print("Failed to gather analysts for {} on attempt {}".format(ticker, tries))
        if tries >= 5:
            print("Too many failed attempts for analysts of {}, skipping to next df.".format(ticker))
            analysis = pd.DataFrame(columns=[ticker])
            break
    
    # Create combined 1D df for later stacking
    combined = pd.concat([summary[ticker].drop(index=['Shares Outstanding']),
                          earnings[ticker],
                          fundies[ticker],
                          valuation[ticker],
                          analysis[ticker]
                         ],
                        axis=0)
    # Remove duplicate rows from combined
    combined = pd.DataFrame(combined.loc[~combined.index.duplicated(keep='first')])
    for analyst in analysis.index:
        combined.loc[analyst+' since'] = analysis.loc[analyst, 'Rating Since']
    # Produce dictionary of results
    results = {'combined':combined, 
               'summary':summary, 
               'earnings':earnings, 
               'earnings_yearly':earnings_yearly, 
               'fundies':fundies, 
               'fundies_yearly':fundies_yearly, 
               'valuation':valuation,
               'analysts':analysis
              }
    return results

def scrape_valuation(driver, ticker, search_first=True):
    """
    This function scrapes the "Valuation" tab of a TD Ameritrade security
    lookup page

    :param driver: (Selenium webdriver) webdriver returned from start_bot()
    :param ticker: (str) ticker symbol to scrape
    :param search_first: (bool) allows for chain of scrapes to be done on one
                                security when set to False. Leave set to True
                                unless you are sure you are already on the
                                desired security, or the wrong data will scrape
    """
    # Search symbol first if flag is True
    if search_first:
        search_symbol(driver, ticker)

    # Find main iframe
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])

    # Switch to Valuation tab
    WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="layout-full"]/nav/ul/li[6]/a')).click()
    #time.sleep(1)

    # Switch to First tab under Valuation (also Valuation)
    WebDriverWait(driver,10).until(lambda x: x.find_element_by_xpath('//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[1]')).click()
    driver.switch_to.default_content()
    iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
    driver.switch_to.frame(iframes[3])

    # Wait for condition before advancing
    WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, '{} vs Industry'.format(ticker)))
    
    # Prepare to scrape valuation tabs by xpath
    tab_names = ['Valuation',
                 'Profitability',
                 'Dividend',
                 'Gowth rates',
                 'Effectiveness',
                 'Financial strength'
                ]
    xpaths = ['//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[1]',
              '//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[2]',
              '//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[3]',
              '//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[4]',
              '//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[5]',
              '//*[@id="stock-valuationmodule"]/div/div[1]/nav/a[6]'
             ]
    tabs = dict(zip(tab_names, xpaths))

    # Scrape each tab
    valuation_df = pd.DataFrame()
    for name, xpath in tabs.items():
        # Switch to Appropriate Report
        driver.find_element_by_xpath(xpath).click()
        iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
        driver.switch_to.frame(iframes[3])
        driver.switch_to.default_content()
        iframes = WebDriverWait(driver, 10).until(lambda x: x.find_elements_by_tag_name("iframe"))
        driver.switch_to.frame(iframes[3])
        WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, '{} vs Industry'.format(ticker)))
        element = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="stock-valuationmodule"]/div/div[1]/div[2]'))
        WebDriverWait(driver, 10).until(lambda x: EC.text_to_be_present_in_element(x, '{} Analysis'.format(name)))
        time.sleep(2)
        # Prevents breaking when there is no info on a tab, by waiting for condition
        try:
            element = driver.find_element_by_xpath('//*[@id="stock-valuationmodule"]/div/div/div[2]/table/tbody/tr[1]/td[2]')
            WebDriverWait(driver, 10).until(lambda x: EC.visibility_of_element_located(element))
        except:
            continue

        # Make soup and find container
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        contain = soup.find('div', {'data-module-name':'StocksValuationModule'})
        
        # Get data
        row_names = [x.get_text() for x in contain.find_all('a', {'class':'definition-link'})]
        tds = soup.find_all('td', {'class':'data-compare'})
        value_dict = {}
        for i, row_name in enumerate(row_names[1:]):
            value_dict[row_name] = {}
            dts = tds[i].find_all('dt')
            dds = tds[i].find_all('dd')
            cols = [dt.get_text() for dt in dts]
            vals = [dd.get_text() for dd in dds]
            value_dict[row_name][cols[0]] = vals[0]
            value_dict[row_name][cols[1]] = vals[1]
            value_dict[row_name]['Type'] = name

        # Create dataframe
        temp = pd.DataFrame.from_dict(value_dict, orient='columns').T
        valuation_df = pd.concat([valuation_df, temp], axis=0, sort=False)
    
    # Clean all columns except 'Type'
    for col in valuation_df.columns:
        if col != 'Type':
            valuation_df[col] = valuation_df[col].apply(lambda x: clean(x))

    # Create ratio to industry feature for normalized feature
    valuation_df['Ratio to Industry'] = valuation_df[ticker] / valuation_df['Industry']
    
    return valuation_df

def scrape_watchlist(driver, tickers, name, root_dir='', skip_finished=True, 
                     save_df=False, errors='ignore', return_skipped=False, ):
    """
    Main wrapper function for scraper. Can do large lists of securities,
    and will store the data into assigned directory (can be set with kwarg)

    :param driver: selenium webdriver
    :param tickers: (list) ticker symbols
    :param name: (str) name of watchlist
    :param save_df: (bool) Whether to save the combined df to disk
    :param errors: (str) 'raise' or 'ignore'
    :param return_skipped: (bool) can return list of skipped securities if
                            ignoring errors

    pass {'root_dir': <user root directory} to change root directory from default
    """
    # Make list for skipped securities if needed
    if return_skipped == True:
        skipped = []

    # Create path name based on date and watchlist name, and make directory
    path_name = root_dir + name + '_' + datetime.today().strftime('%m-%d-%Y')
    if not os.path.isdir(path_name):
        os.mkdir(path_name)
    
    # Create empty dataframe
    big_df = pd.DataFrame()
    
    # Scrape each ticker
    for i, ticker in enumerate(tickers):
        tickers_done = i + 1
        # Establish ticker path
        ticker_path = path_name+'/{}'.format(ticker)
        
        # Skip previously scraped securities if flag is True
        if skip_finished:
            if os.path.isdir(ticker_path):
                continue
        
        # Scrape security
        try:
            results = scrape_ticker(driver, ticker)
        except:
            print("Did not successfully scrape {}".format(ticker))
            if errors == 'raise':
                raise
            else:
                if return_skipped:
                    skipped.append(ticker)
                continue
        
        # Make directory if there is none
        if not os.path.isdir(ticker_path):
            os.mkdir(ticker_path)
        
        # Dump .csv files to directory
        for name, dataframe in results.items():
            try:
                dataframe.to_csv(ticker_path + '/{}'.format(name) + '.csv')
            except:
                print("No {} dataframe for {}".format(name,ticker))
        
        # Compile security to big_df
        big_df = pd.concat([big_df, results['combined'].T], axis=0, sort=True)
        
        # Print number of tickers completed every 10 completions
        if tickers_done % 10 == 0:
            print("{} tickers scraped".format(tickers_done))

    # Saves combined dataframe to file if called
    if save_df:
        big_df.to_csv(path_name + '/{}'.format('big_df.csv'))
    
    if not return_skipped:
        return big_df,
    else:
        return big_df, skipped