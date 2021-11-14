from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import datetime
import FinanceDataReader as fdr

def loadStockData(symbol, startDate, endDate):
    df_stock = fdr.DataReader(symbol, startDate.isoformat(), endDate.isoformat())
    df_stock = df_stock[['Close']]
    df_stock['Fluctuation'] = df_stock['Close'].div(df_stock['Close'].shift(1)).apply(lambda x : (x - 1) * 100)
    return df_stock

def loadArticleData(url):
    resp = requests.get(url)

    titles = []
    links = []
    pubDates = []
    descriptions = []

    for item in bs(resp.text, 'xml').find_all('item'):
        title = item.title.string 
        source = item.source.string # 언론사
        titles.append(title[:title.find(source) - 3]) 
        # Possilbe problem: what if title contains the source at the very beginning?

        links.append(item.link.string)
        pubDates.append(item.pubDate.string)

        # Description block has below format:
        # <a href="https://..." target="_blank">2.3m ... - ...</a>&nbsp;&nbsp;<font color="#6f6f6f">...</font>
        descriptions.append(item.description.string.split('>')[1].split('<')[0])

    data = {'title': titles, 'link': links, 'pubDate': pubDates, 'description': descriptions}
    return pd.DataFrame(data, columns=['title', 'link', 'pubDate', 'description']), len(titles)

def crawl(companyName, startDate, endDate, isKor=True): 
    if isKor:
        country = ('ko', 'KR')
        symbol = str(df_kospi.loc[df_kospi['Name'] == companyName]['Symbol'].values[0])
        print(f'Start crawling for {companyName} in Google News Korea')
    else:
        country = ('en', 'US')
        symbol = df_snp.loc[df_snp['Name'] == companyName]['Symbol'].values[0]
        print(f'Start crawling for {companyName} in Google News US')

    df_stock = loadStockData(symbol, startDate - datetime.timedelta(days=1), endDate)
    df_stock.to_csv(f'./stock/{country[1]}/{companyName}_{startDate.isoformat()}_{endDate.isoformat()}.csv')
    print(f'Loaded {companyName} price info, from {startDate.isoformat()} to {endDate.isoformat()}!')
    
    currentDate = startDate
    totalCount = 0
    while currentDate <= endDate:
        nextDate = currentDate + datetime.timedelta(days=1)
        url = f'https://news.google.com/rss/search?q={companyName}+after:{currentDate.isoformat()}+before:{nextDate.isoformat()}& \
                hl={country[0]}&gl={country[1]}&ceid={country[1]}:{country[0]}'
        df_articles, dailyCount = loadArticleData(url)
        totalCount += dailyCount
        df_articles.to_csv(f'./news/{country[1]}/{companyName}_{currentDate.isoformat()}.csv')
        print(f'  {currentDate.isoformat()}: {dailyCount} articles')
        currentDate = nextDate
    print(f'Crawled {totalCount} articles in total!')

if __name__=="__main__":
    df_kospi = fdr.StockListing('KOSPI')
    df_snp = fdr.StockListing('S&P500')
    # May replace w/ fixed dictionary

    startDate = datetime.date(2020, 3, 3) # inclusive
    endDate = datetime.date(2020, 3, 10) # inclusive
    companyListK = ['삼성전자', '한국조선해양', '신세계']
    for companyName in companyListK:
        crawl(companyName, startDate, endDate)

    companyListUS = ['Apple', 'IBM', 'Delta Air Lines']
    for companyName in companyListUS:
        crawl(companyName, startDate, endDate, False)