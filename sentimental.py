from google.colab import drive
drive.mount('/content/drive')

!pip3 install yfinance

import yfinance as yf

'''
Class that gets stock data based on the ticker symbol passed in
Data clan be queried by specifying period and interval
period = 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval = 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
'''
class StockData:

    def __init__(self, ticker="TSLA"):
        self.stock = yf.Ticker(ticker)

    # ohlc = Open, High, Low, Close, or All
    def getPrice(self, period="1mo", interval="1d", ohlc="All"):
        data = self.stock.history(period=period, interval=interval)

        if ohlc== "All":
            return data.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)

        # parse out other columns except ohlc data
        return data[ohlc]

    def getVolume(self, period="1mo", interval="1d"):
        data = self.stock.history(period=period, interval=interval)
        # parse out uneeded columns
        return data.filter(['Date','Volume'])
        
!pip3 install searchtweets      
      
      
import re
import requests
import pandas as pd
from searchtweets.api_utils import convert_utc_time


class Twitter:
    def __init__(self, config):
        self.key = config['api_config']['twitter']
        self.base_url = 'https://api.twitter.com/2/tweets/search/all'

    # date must be in YYYY-MM-DDTHH:mm:ss
    def get_tweets(self, ticker, date_since='2019-04-15T00:00:00', date_until='2021-04-16T00:00:00', limit=100):
        headers = self.create_headers()
        params = self.create_params(ticker, date_since, date_until, limit)
        response = requests.request('GET', self.base_url, headers=headers, params=params).json()
        # data to pandas df
        print(response)
        # df = pd.DataFrame.from_records(response['data'])
        # # clean data
        # df['clean_text'] = df['text'].apply(self.clean_tweet)
        return df

    def clean_tweet(self, tweet):
        # remove @ mentions, hashtags, Re-tweets, Hyper Links, and new lines
        tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub(r'RT[\s]+', '', tweet)
        tweet = re.sub(r'https?:\/\/\S+', '', tweet)
        tweet = re.sub(r'(\r\n|\n|\r)', '', tweet)
        return tweet

    def create_params(self, ticker, date_since, date_until, limit):
        # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
        # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
        query_params = {
            'query': f'(from:${ticker} -is:retweet) OR #{ticker}',
            'tweet.fields': 'id,created_at,text',
            'start_time': date_since,
            'end_time': date_until,
            'max_results': limit
        }
        return query_params

    def create_headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    def utc_convert(self, time):
        pass

      
# Data imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # ML imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
      
      
# read data
df = pd.read_csv("/content/drive/MyDrive/ds340W/data/tweet_data.csv")

# sample data
df = df.sample(frac=1).reset_index(drop=True)

# clean tweet text
df['Text'] = df['Text'].apply(lambda x: x.lower())  # transform text to lowercase
df['Text'] = df['Text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
print(df.shape)
df.head(10)
      
df['Sentiment'].value_counts().sort_index().plot.bar()     
      
df['Text'].str.len().plot.hist()



# X as tokenize data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Text'].values)
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X)
print("X tokenized data = ", X[:5])


# Y as buckets of Sentiment column
y = pd.get_dummies(df['Sentiment']).values
[print(df['Sentiment'][i], y[i]) for i in range(0, 5)]


# lstm create model
model = Sequential()
model.add(Embedding(5000, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(GRU(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

batch_size = 32
epochs = 8

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

predictions = model.predict(X_test)

pos_count, neg_count = 0, 0
real_pos, real_neg = 0, 0
for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==1:
        pos_count += 1
    else:
        neg_count += 1
    if np.argmax(y_test[i])==1:    
        real_pos += 1
    else:
        real_neg +=1

print('Positive predictions:', pos_count)
print('Negative predictions:', neg_count)

print('Real neutral:', real_pos)
print('Real negative:', real_neg)

import matplotlib.pyplot as plt

print(history.history['loss'], )
predictions = [pos_count, neg_count]
real = [real_pos, real_neg]
labels = ['Positive', 'Negative']

x = np.arange(len(labels))
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, real, width, label='Real')
rects2 = ax.bar(x + width/2, predictions, width, label='Predictions')

ax.set_ylabel('Scores')
ax.set_title('Count of Classifications')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
loss = history.history['loss']
epoch = [item for item in range(1,9)]
accuracy = history.history['accuracy']
ax.plot(epoch, loss, label = "Loss")
ax.plot(epoch, accuracy, label = "Accuracy")

ax.set_xlabel('Epoch')
ax.set_title('Accuracy and Loss per epoch')
plt.legend()
plt.show()
