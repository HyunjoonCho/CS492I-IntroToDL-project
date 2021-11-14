from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import requests
import datetime

def loadData(driver, url):
    driver.get(url)
    driver.implicitly_wait(3)
    url = driver.current_url
    resp = requests.get(url)
    return bs(resp.text, 'xml')

def parseToDataFrame(soup):
    titles = []
    links = []
    pubDates = []
    descriptions = []

    for item in soup.find_all('item'):
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
        print(f'Start crawling for {companyName} in Google News Korea')
    else:
        country = ('en', 'US')
        print(f'Start crawling for {companyName} in Google News US')
    
    currentDate = startDate
    totalCount = 0
    while currentDate != endDate:
        nextDate = currentDate + datetime.timedelta(days=1)
        url = f'https://news.google.com/rss/search?q={companyName}+after:{currentDate.isoformat()}+before:{nextDate.isoformat()}& \
                hl={country[0]}&gl={country[1]}&ceid={country[1]}:{country[0]}'
        soup = loadData(driver, url)
        df, dailyCount = parseToDataFrame(soup)
        totalCount += dailyCount
        df.to_csv(f'./news/{country[1]}/{companyName}_{currentDate.isoformat()}.csv')
        print(f'  {currentDate.isoformat()}: {dailyCount} articles')
        currentDate = nextDate
    print(f'Crawled {totalCount} articles in total!')

if __name__=="__main__":
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    driver = webdriver.Chrome('chromedriver', options=options)

    startDate = datetime.date(2020, 3, 3) # inclusive
    endDate = datetime.date(2020, 3, 10) # exclusive
    companyListK = ['삼성전자', '한국조선해양', '신세계']
    for companyName in companyListK:
        crawl(companyName, startDate, endDate)

    companyListUS = ['Apple', 'IBM', 'Delta Air Lines']
    for companyName in companyListUS:
        crawl(companyName, startDate, endDate, False)