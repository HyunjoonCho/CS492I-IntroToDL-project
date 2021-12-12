import streamlit as st
import FinanceDataReader as fdr
from bs4 import BeautifulSoup as bs
import requests
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import web_helper

st.title('Investement Advisor')
df_kospi = fdr.StockListing('KOSPI')
fluctuation_category = ['Big Drop...', 'Slight Drop..', 'Slight Jump!', 'Big Jump!!']
helper = web_helper.WebHelper()

def get_ticker(company_name):
    return str(df_kospi.loc[df_kospi['Name'] == company_name]['Symbol'].values[0])

def get_last_traiding_day(stock_price_data):
    return stock_price_data.index.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')).values[-1]

@st.cache
def load_stock_price_data(ticker, predict_date):
    df_stock = fdr.DataReader(ticker, predict_date - datetime.timedelta(days=40), predict_date- datetime.timedelta(days=1))
    df_stock = df_stock[-20:]
    df_closing = df_stock[['Close']]
    return df_closing

@st.cache
def load_news_data(company_name, last_trading_day, predict_date):
    url = f'https://news.google.com/rss/search?q={company_name}+after:{last_trading_day}+before:{predict_date}& \
            hl=ko&gl=KR&ceid=KR:ko'
    resp = requests.get(url)
    articles = []

    for item in bs(resp.text, 'xml').find_all('item'):
        title = item.title.string
        link = item.link.string
        source = item.source.string # 언론사
        if helper.filter_by_category(company_name, title):
            articles.append((title[:title.find(source) - 3], link))

    return articles

def predict_on_news(articles):
    aggregated_titles = ' '.join([headline for headline, _ in articles])
    return helper.predict_fluctuation_by_news(aggregated_titles)

def predict_on_chart(ticker, stock_price_data):
    return helper.predict_fluctuation_by_chart(ticker, stock_price_data)

company_name = st.text_input('Company Name', '삼성전자')
year = int(st.text_input('Year', '2021'))
month = int(st.text_input('Month', '12'))
date = int(st.text_input('Date', '13'))
predict_date = datetime.date(year, month, date)

data_load_state = st.text('Loading data...')
stock_price_data = load_stock_price_data(get_ticker(company_name), predict_date)
articles = load_news_data(company_name, get_last_traiding_day(stock_price_data), predict_date)
data_load_state.text("Done! (using st.cache)")

st.subheader(f'{company_name} on {predict_date.isoformat()}')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Related Articles')
    if len(articles) < 10:
        for title, link in articles:
            st.write(f'[{title}]({link})')
    else:
        for title, link in articles[:10]:
            st.write(f'[{title}]({link})')
        with st.expander("More Articles"):
            for title, link in articles[10:]:
                st.write(f'[{title}]({link})')

with col2:
    st.subheader('20 Days Closing Price')
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    ax = sns.lineplot(x = stock_price_data.index, y = stock_price_data['Close'], label="Closing Price", color='tomato')
    ax.set_title('Stock Price', size = 14, fontweight='bold')
    ax.set_xlabel('Days', size = 12)
    ax.set_ylabel('Price(won)', size = 14)
    st.pyplot(fig)
    if st.checkbox('Show raw price data'):
        st.subheader('Raw Price Data')
        st.write(stock_price_data)

    st.subheader('Our Prediction')
    prediction_state = st.text('Now Predicting...')
    chart_pred, predicted_price = predict_on_chart(get_ticker(company_name), stock_price_data)
    news_pred, news_confidence = predict_on_news(articles)
    prediction_state.text('Prediction Result')
    st.write(f'Based on Chart, **{fluctuation_category[chart_pred]}**')
    st.write(f'To be more specific, **{predicted_price}won**')    
    st.write(f'Based on Article, **{fluctuation_category[news_pred]}**')
    st.write(f'We are **{news_confidence:.2f}%** confident')