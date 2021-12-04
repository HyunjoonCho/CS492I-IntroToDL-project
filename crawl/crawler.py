from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import datetime
import FinanceDataReader as fdr
import OpenDartReader as odr

def loadStockData(symbol, startDate, endDate):
    df_stock = fdr.DataReader(symbol, startDate.isoformat(), endDate.isoformat())
    df_stock = df_stock[['Close']]
    df_stock['Fluctuation'] = df_stock['Close'].div(df_stock['Close'].shift(1)).apply(lambda x : (x - 1) * 100)
    return df_stock

def aggregateTitles(companyName, url):
    resp = requests.get(url)
    titles = []

    for item in bs(resp.text, 'xml').find_all('item'):
        title = item.title.string
        source = item.source.string # 언론사
        titles.append(title[:title.find(source) - 3])

    return ' '.join(titles)

def classifyFluctuation(fluctuation):
    if fluctuation < -2.5:
        return 0
    elif fluctuation < 0:
        return 1
    elif fluctuation < 2.5:
        return 2
    else:
        return 3

def crawl_news(companyName, startDate, endDate, isKor=True): 
    if isKor:
        country = ('ko', 'KR')
        symbol = str(df_kospi.loc[df_kospi['Name'] == companyName]['Symbol'].values[0])
        print(f'Start crawling for {companyName} in Google News Korea')
    else:
        country = ('en', 'US')
        symbol = df_snp.loc[df_snp['Name'] == companyName]['Symbol'].values[0]
        print(f'Start crawling for {companyName} in Google News US')

    df_stock = loadStockData(symbol, startDate - datetime.timedelta(days=1), endDate)
    # df_stock.to_csv(f'./stock/{country[1]}/{companyName}_{startDate.isoformat()}_{endDate.isoformat()}.csv')
    print(f'Loaded {companyName} price info, from {startDate.isoformat()} to {endDate.isoformat()}!')

    dateList = df_stock.index.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')).values
    fluctuationList = df_stock.loc[:, 'Fluctuation'].values

    idx = 1
    while idx < len(dateList):
        url = f'https://news.google.com/rss/search?q={companyName}+after:{dateList[idx - 1]}+before:{dateList[idx]}& \
                hl={country[0]}&gl={country[1]}&ceid={country[1]}:{country[0]}'
        aggTitle = aggregateTitles(companyName, url)
        if aggTitle:
            with open(f'./exp/{classifyFluctuation(fluctuationList[idx])}/{companyName}_{dateList[idx]}.txt', 
                        'w', encoding='UTF-8') as file:
                file.write(aggTitle)
        idx += 1

def crawl_disclosure_k(companyName, startDate, endDate):
    print(f'Start crawling for {companyName} in DART')
    api_key = '' # create your own api key then paste
    dart = odr(api_key)
    df_disclosure = dart.list(companyName, start=startDate, end=endDate)
    if not df_disclosure.empty:
        df_disclosure = df_disclosure[['report_nm', 'rcept_no']]
        df_disclosure = df_disclosure[~df_disclosure['report_nm'].str.contains('기재정정')] # TODO: exclude or not?
        df_disclosure['rcept_no'] = df_disclosure['rcept_no'].apply(lambda x : datetime.date(int(x[:4]), int(x[4:6]), int(x[6:8])))

        symbol = str(df_kospi.loc[df_kospi['Name'] == companyName]['Symbol'].values[0])
        df_stock = loadStockData(symbol, startDate - datetime.timedelta(days=1), endDate)
        print(f'Loaded {companyName} price info, from {startDate.isoformat()} to {endDate.isoformat()}!')

        dateList = df_stock.index.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')).values
        fluctuationList = df_stock.loc[:, 'Fluctuation'].values
        dailyFluctuationDict = dict(zip(dateList, fluctuationList))

if __name__=="__main__":
    df_kospi = fdr.StockListing('KOSPI')
    df_snp = fdr.StockListing('S&P500')
    # May replace w/ fixed dictionary

    startDate = datetime.date(2020, 3, 3) # inclusive
    endDate = datetime.date(2020, 4, 16) # inclusive
    companyListK = ['삼성전자', 'SK하이닉스', 'NAVER', '삼성바이오로직스', '카카오', 'LG화학', '삼성SDI', 
                    '현대차', '기아', '셀트리온', '카카오뱅크', '크래프톤', 'POSCO', 'KB금융', '현대모비스', 
                    '카카오페이', '삼성물산', 'SK이노베이션', 'LG전자', '신한지주', 'LG생활건강', 'SK바이오사이언스', 
                    '하이브', '엔씨소프트', '한국전력', '삼성생명', '두산중공업', '하나금융지주', 'HMM', '삼성전기', 
                    '삼성에스디에스', 'SK아이이테크놀로지', 'KT&G', '넷마블', '포스코케미칼', '아모레퍼시픽', '삼성화재', 
                    '대한항공', 'S-Oil', '우리금융지주', '현대중공업', '고려아연', '기업은행', 'KT', 'SK바이오팜', 'LG디스플레이', '한온시스템']
    # 우리금융지주 수집 중 "Remote end closed connection without" urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    # KOSPI 시총 상위 50개 종목, 지주회사 제외

    for companyName in companyListK:
        crawl_news(companyName, startDate, endDate)

    # companyListUS = ['Apple', 'IBM', 'Delta Air Lines']
    # for companyName in companyListUS:
    #     crawl(companyName, startDate, endDate, False)