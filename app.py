from flask import Flask, render_template, request, jsonify
import plotly.express as px
from markupsafe import Markup
from pandas import Timestamp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from transformers import pipeline
from ipywidgets import interact, widgets
from ipywidgets import interactive
from IPython.display import HTML, display, clear_output
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import HTML
import json
from dateutil import parser, tz
from transformers import BertTokenizer, BertForTokenClassification
import torch
import pyttsx3
from msedge.selenium_tools import EdgeOptions, Edge


app = Flask(__name__)
stock_sentiment_dataframe = None
search_query = ""
last_fetch_time = None
news_alert_df = pd.DataFrame()


def sentiment_converter(news_dataframe):
    model = pipeline('sentiment-analysis')
    
#     news_dataframe["headline_sentiment"] = news_dataframe.Headline.apply(model)
#     news_dataframe.headline_sentiment = news_dataframe.headline_sentiment.apply(lambda x: -x[0]['score'] if x[0]['label'] == 'NEGATIVE' else x[0]['score'])
    
    news_dataframe["summary_sentiment"] = news_dataframe.Summary.apply(model)
    news_dataframe.summary_sentiment = news_dataframe.summary_sentiment.apply(lambda x: -x[0]['score'] if x[0]['label'] == 'NEGATIVE' else x[0]['score'])
    
#     news_dataframe['secs_ago'] = news_dataframe.Datetime.apply(secs_converter)
    
    return news_dataframe

def convert_to_datetime(value):
    if 'minute' in value:
        return datetime.now() - timedelta(minutes=float(value.split()[0]))
    elif 'hour' in value:
        return datetime.now() - timedelta(hours=float(value.split()[0]))
    elif 'day' in value:
        return datetime.now() - timedelta(days=float(value.split()[0]))
    else:
        return None
    
def update_plot(interval_minutes, df, search_query):
    clear_output(wait=True)

    if interval_minutes is None:
        interval_minutes = 1
    else:
        interval_minutes = int(interval_minutes)

    query_searched = ""
    if search_query is not None:
        query_searched = search_query
    else:
        query_searched = ""

    if interval_minutes == 1:
        time_range = pd.to_timedelta('1H')
    elif interval_minutes == 3:
        time_range = pd.to_timedelta('3H')
    elif interval_minutes == 5:
        time_range = pd.to_timedelta('6H')
    else:
        time_range = pd.to_timedelta('24H')

    # Resample the DataFrame based on the time interval and calculate the mean of 'summary_sentiment'
    # df_resampled = df.resample(f'{interval_minutes}T').mean().last(time_range)
    df_resampled = df['summary_sentiment'].resample(f'{interval_minutes}T').mean().last(time_range)
    df_resampled = pd.DataFrame(df_resampled, columns=['summary_sentiment'])

    # Extract time from the timestamp
    df_resampled['Time'] = df_resampled.index.time

    # Format the x-axis labels to show both date and time at midnight,
    # and show only time for other intervals
    df_resampled['x_labels'] = [
        f'{date.strftime("%Y-%m-%d %H:%M")}' if (time >= pd.Timestamp('00:00:00').time() and time <= pd.Timestamp('00:30:00').time()) else f'{time.strftime("%H:%M")}'
        for date, time in zip(df_resampled.index, df_resampled['Time'])
    ]

    # Set color based on sentiment directly without a color scale
    df_resampled['colors'] = ['green' if sentiment >= 0 else 'red' for sentiment in df_resampled['summary_sentiment']]

    # Create a Plotly figure with a consistent size
    fig = go.Figure()

    # Add stacked bar traces
    fig.add_trace(go.Bar(
        x=df_resampled.index,
        y=df_resampled['summary_sentiment'],
        marker_color=df_resampled['colors'],
        hovertemplate='Datetime: %{x|%Y-%m-%d %H:%M}<br>Mean Sentiment: %{y:.4f}',
    ))

    # Set x-axis labels and tilt them to the left
    fig.update_layout(
        title=f'{query_searched} - Recent news sentiment {interval_minutes}-Minute Interval'
        # xaxis_title='Datetime',
    )
    print("inside update_plot = ", interval_minutes)
    # Calculate tickvals and ticktext
    if interval_minutes != 0:
        tickvals = np.linspace(0, len(df_resampled) - 1, num=len(df_resampled) // max(1, int(30 / interval_minutes)), endpoint=True, dtype=int)
        ticktext = [pd.Timestamp(x).ceil('30T').strftime('%Y-%m-%d %H:%M') for x in df_resampled.index[tickvals]]
    else:
        tickvals = np.arange(0, len(df_resampled), dtype=int)
        ticktext = [df_resampled['x_labels'].iloc[i] for i in tickvals]

    # Set x-axis ticks
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=df_resampled.index[::int(60 / interval_minutes)],
            ticktext=df_resampled['x_labels'][::int(60 / interval_minutes)],
        
#         tickvals=df_resampled.index[tickvals].tolist(),
#         ticktext=ticktext,
            tickangle=-45,  # Tilt the labels to the left
        ),
    )

    # Remove the color legend and color scale
    fig.update_layout(
        showlegend=False,
    )

    # Consistent figure size
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
    )

    # Show the plot
    # fig.show(renderer='notebook_connected')
    # fig.show()

    df.style.set_table_styles([{
        'selector': 'thead th',
        'props': [('text-align', 'center')]
    }])

    # tail_html = df.tail().to_html(classes='table table-striped', index=True)
    plot_html = fig.to_html(full_html=False)

    return plot_html

def fetch_news_money_control(url, search_query = ""):
        # display(search_query)
    if len(search_query) == 0:
        edge_driver_path = './msedgedriver.exe'  # Change this to the path of your Edge WebDriver

        # Set up Edge options for headless mode
        edge_options = EdgeOptions()
        edge_options.use_chromium = True  # Use Chromium-based Edge
        edge_options.add_argument('--headless')

        # Create a WebDriver instance with the specified options
        driver = Edge(options=edge_options, executable_path=edge_driver_path)

        # chrome_driver_path = './chrome.exe'
        # driver = webdriver.Edge(executable_path=edge_driver_path)
        # driver = webdriver.Chrome(executable_path=chrome_driver_path)
        driver.get(url)

        def convert_to_datetime(df, datetime_column):
        # Apply datetime parsing
            df[datetime_column] = df[datetime_column].apply(lambda x: parser.parse(x, fuzzy=True) if pd.notnull(x) else None)

            # Strip the last +05:30 part and format in 24-hour format
            df[datetime_column] = df[datetime_column].dt.strftime('%B %d, %Y %H:%M')

            # Convert to datetime64[ns]
            df[datetime_column] = pd.to_datetime(df[datetime_column])

            return df

        search_results_html = driver.page_source
        soup = BeautifulSoup(search_results_html, 'html.parser')

        news_elements = soup.find_all('li', class_='clearfix')

        if len(news_elements) > 0 :  
            headlines = []
            summaries = []
            date_times = []
            article_urls = []

            for news_element in news_elements:
                headline = news_element.find('h2').text
                summary = news_element.find('p').text
                datetime_element = news_element.find('span')
                date_time = datetime_element.text.strip()
                article_url = news_element.find('a')['href']

                headlines.append(headline)
                summaries.append(summary)
                date_times.append(date_time)
                article_urls.append(article_url)

                # Create a DataFrame
                news_df = pd.DataFrame({
                'Source' : 'Money Control',
                'Headline': headlines,
                'Summary': summaries,
                'Datetime': date_times,
                'URL': article_urls
                })

                news_df = convert_to_datetime(news_df, 'Datetime')
            # news_df['ArticleContent'] = news_df.URL.apply(fetch_paragraph_content)
            driver.quit()
            return news_df
        else:
            driver.quit()
            return None
        
def fetch_news_zerodha(url, search_query = ""):

    edge_driver_path = './msedgedriver.exe'
    
    # Set up Edge options for headless mode
    edge_options = EdgeOptions()
    edge_options.use_chromium = True  # Use Chromium-based Edge
    edge_options.add_argument('--headless')

    # Create a WebDriver instance with the specified options
    driver = Edge(options=edge_options, executable_path=edge_driver_path)

    driver.get(url)

    def convert_to_datetime(value):
        if 'minute' in value:
            return datetime.now() - timedelta(minutes=float(value.split()[0]))
        elif 'hour' in value:
            return datetime.now() - timedelta(hours=float(value.split()[0]))
        elif 'day' in value:
            return datetime.now() - timedelta(days=float(value.split()[0]))
        else:
            return None


    search_bar = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//input[@id="q"]'))
    )  
    
    search_bar.send_keys(search_query)
    time.sleep(5) 
    search_results_html = driver.page_source
    soup = BeautifulSoup(search_results_html, 'html.parser')

    news_elements = soup.find_all('li', class_='box item')
    if len(news_elements) > 0 :  
        headlines = []
        summaries = []
        date_times = []
        article_urls = []

        for news_element in news_elements:
            headline = news_element.find('h2').text
            summary = news_element.find('div').text
            datetime_element = news_element.find('span', class_='date')
            date_time = datetime_element.text if datetime_element else 'Date and time not available'
            article_url = news_element.find('a')['href']

            headlines.append(headline)
            summaries.append(summary)
            date_times.append(date_time)
            article_urls.append(article_url)

          # Create a DataFrame
            news_df = pd.DataFrame({
            'Source' : 'Zerodha',
            'Headline': headlines,
            'Summary': summaries,
            'Datetime': date_times,
            'URL': article_urls
          })
            
        news_df['Datetime'] = news_df['Datetime'].apply(convert_to_datetime)
        driver.quit()
        return news_df
    else:
        driver.quit()
        return None



def all_platform_news_fetch(url_lst, search_query = ""):
    news_df_zerodha = pd.DataFrame()
    news_df_money_control = pd.DataFrame()
    all_news_df = pd.DataFrame()
    for url in url_lst:
        if url == "https://pulse.zerodha.com/":
            news_df_zerodha = fetch_news_zerodha(url, search_query)
        elif url == "https://www.moneycontrol.com/news/business/markets/":
            news_df_money_control = fetch_news_money_control(url, search_query)

    if not ((news_df_zerodha is None) and (news_df_money_control is None)):
        all_news_df = pd.concat([news_df_zerodha, news_df_money_control], ignore_index=True)
        all_news_df = all_news_df.sort_values(by='Datetime', ascending=False)
        all_news_df = all_news_df.reset_index(drop=True)
        return all_news_df
    else:
        return None

def extract_stocks_from_text(text):
    # Load pre-trained BERT model and tokenizer for NER
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    # Tokenize input
    tokens = tokenizer(text, return_tensors='pt', truncation=True)

    # Make a prediction
    outputs = model(**tokens)

    # Get predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2)

    # Map labels back to entities
    predicted_entities = tokenizer.batch_decode(tokens['input_ids'][0], skip_special_tokens=True)
    entity_labels = [model.config.id2label[label_id] for label_id in predicted_labels[0].tolist()]

    # Extract entities (stock symbols)
    stocks_impacted = [entity for entity, label in zip(predicted_entities, entity_labels) if label != 'O']

    # Clean and merge
    cleaned_stocks = []
    current_stock = ""

    for word in stocks_impacted:
        if word.startswith('#'):
            current_stock += word.replace(' ', '').replace('#', '')
        else:
            if current_stock:
                cleaned_stocks.append(current_stock)
                current_stock = ""
            current_stock += word.replace(' ', '').replace('#', '')

    # Add the last stock if any
    if current_stock:
        cleaned_stocks.append(current_stock)

    # Join the cleaned stocks with a comma
    result = ', '.join(cleaned_stocks)

    return result


    
def monitor_website():
    global news_alert_df
    print("Monitor website function called.")
    url1 = "https://www.moneycontrol.com/news/business/markets/"
    url2 = "https://pulse.zerodha.com/"
    url_lst =[url1 ,url2]
    
    news_df = all_platform_news_fetch(url_lst)
    current_time = datetime.now()

    # Set the time threshold for the last 10 minutes
    time_threshold = current_time - timedelta(minutes=10)
    news_df = news_df[news_df['Datetime'] > time_threshold]

    # news_df['stocks_impacted'] = news_df['Summary'].apply(extract_stocks_from_text)
    news_df['stocks_impacted'] = news_df['Summary'].apply(extract_stocks_from_text) + ', ' + news_df['Headline'].apply(extract_stocks_from_text)
    news_df = sentiment_converter(news_df)

    if news_df.empty:
        print("No updates")
    else:
        if news_alert_df.equals(news_df):
            pass
        else:
            if news_alert_df.empty:
                speak("News update found")
                news_alert_df = news_df
                # display(news_df)
            else:
                if news_alert_df['Headline'].iloc[0] == news_df['Headline'].iloc[0]:
                    pass
                else:
                    speak("News update found")
                    news_alert_df = news_df
                    # display(news_df)


    # rows = 4
    # data = {
    #     'Name': [f'Person {i}' for i in range(1, rows + 1)],
    #     'Age': [random.randint(20, 40) for _ in range(rows)],
    #     'City': [random.choice(['New York', 'San Francisco', 'Los Angeles', 'Chicago']) for _ in range(rows)]
    # }
    # news_df = pd.DataFrame(data)

    # Print the updated DataFrame
    print(news_df)
    print("datetime is = ", news_df.Datetime)
    return news_df

def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/news_alert')
def news_alert():
    # Your existing code for rendering the template
    global news_alert_df
    return render_template('news_alert.html', news_alert_df=news_alert_df, current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/get_dataframe')
def get_dataframe():
    global news_alert_df
    
    news_alert_df = monitor_website()
    display("moniter website fun triggered")

    news_alert_df['Datetime'] = news_alert_df['Datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    column_order = ['Datetime', 'stocks_impacted', 'summary_sentiment'] + [col for col in news_alert_df.columns if col not in ['Datetime', 'stocks_impacted', 'summary_sentiment']]
    news_alert_df = news_alert_df[column_order]

    df_json = news_alert_df.to_json(orient='split')
    
    # Print the JSON data
    print("JSON Data:", df_json)
    
    return jsonify({'data': df_json})


@app.route('/get_datetime')
def get_datetime():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'datetime': current_datetime})


@app.route('/stock_sentiment_analysis', methods=['GET', 'POST'])
def stock_sentiment_analysis():
    global stock_sentiment_dataframe  # Reference the global DataFrame
    global search_query
    global last_fetch_time

    interval_dropdown = [1, 3, 5, 15, 30, 60]
    default_value = ""
    url1 = "https://www.moneycontrol.com/news/business/markets/"
    url2 = "https://pulse.zerodha.com/"

    url_lst =[url1 ,url2]

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'fetch_sentiment':
            # Fetch news data using the search_query
            search_query = request.form.get('search_query')
            interval_minutes = (request.form.get('interval_dropdown'))

            news_df = all_platform_news_fetch(url_lst, search_query)

            if news_df is None:
                return "No news found"
            else:
                # Add the sentiment conversion and datetime conversion steps if needed
                news_df = sentiment_converter(news_df)
                # news_df['Datetime'] = news_df['Datetime'].apply(convert_to_datetime)
                news_df = news_df.sort_values(by='Datetime')

                news_df['Datetime'] = pd.to_datetime(news_df['Datetime'])
                news_df.set_index('Datetime', inplace=True)

                # Update the global DataFrame
                stock_sentiment_dataframe = news_df

                last_fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Last Fetched = ", last_fetch_time)

                # Update the plot using the new news_df
                plot_html = update_plot(interval_minutes, news_df, search_query)

                # Generate HTML for DataFrame tail
                tail_html = news_df.tail().iloc[::-1].to_html(classes='table table-bordered', col_space=100, justify='center')

                return render_template('stock_sentiment_analysis.html', options=interval_dropdown, default_value = default_value, plot_html=plot_html,
                                       search_query=search_query, tail_html=tail_html, last_fetch_time = last_fetch_time)

            
        else:
            print("inside interval change")
            # If the interval dropdown is changed, update the plot using the existing global DataFrame
            # interval_minutes = int(request.json.get('interval_dropdown', default_value))
            interval_minutes = (request.form.get('interval_dropdown'))
            print("inside else = ", interval_minutes)

            if stock_sentiment_dataframe is not None:
                print("stock_sentiment_dataframe is not none")
                # Update the plot using the existing global DataFrame
                plot_html = update_plot(interval_minutes, stock_sentiment_dataframe, search_query)
                tail_html = stock_sentiment_dataframe.tail().iloc[::-1].to_html(classes='table table-bordered', col_space=100, justify='center')

                # Return the updated plot content as JSON using Flask's jsonify

                return render_template('stock_sentiment_analysis.html', options=interval_dropdown, default_value = default_value, plot_html=plot_html,
                                       search_query=search_query, tail_html=tail_html, last_fetch_time = last_fetch_time)
            else:
                # Handle the case where stock_sentiment_dataframe is None
                print("stock_sentiment_dataframe is none")
                return 'error: Global DataFrame is not available'


    # Default rendering for GET requests or other cases
    return render_template('stock_sentiment_analysis.html', options=interval_dropdown, default_value = default_value, plot_html=None,
                           search_query=search_query, tail_html=None, last_fetch_time = last_fetch_time)



if __name__ == '__main__':
    app.run(debug=True)
