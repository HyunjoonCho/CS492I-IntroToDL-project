from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import requests
import datetime

def crawl(companyName, isKor=True): 
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    driver = webdriver.Chrome('chromedriver', options=options)

    if isKor:
        country = ('ko', 'KR')
        print(f'Start crawling for {companyName} in Google News Korea')
    else:
        country = ('en', 'US')
        print(f'Start crawling for {companyName} in Google News US')

    url = f'https://news.google.com/rss/search?q={companyName}+when:1d&hl={country[0]}&gl={country[1]}&ceid={country[1]}:{country[0]}'
    driver.get(url)
    driver.implicitly_wait(3)
    url = driver.current_url
    resp = requests.get(url)
    soup = bs(resp.text, 'xml')

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
    data_frame = pd.DataFrame(data, columns=['title', 'link', 'pubDate', 'description'])
    data_frame.to_csv(f'./news/{country[1]}/{companyName}.csv')
    print(f'Crawled {len(titles)} articles!')

if __name__=="__main__":
    companyListK = ['삼성전자', '한국조선해양', '하이트진로', '쿠콘']
    for companyName in companyListK:
        crawl(companyName, True)

    companyListUS = ['Apple', 'IBM', 'Delta Air Lines']
    for companyName in companyListUS:
        crawl(companyName, False)