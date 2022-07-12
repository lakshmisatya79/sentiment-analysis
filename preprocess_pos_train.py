import re
import io
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import csv
# porter = PorterStemmer()
import sys
lemmatizer=WordNetLemmatizer()
#lancaster = LancasterStemmer()
# import utils
# from nltk.stem import PorterStemmer
stop_words1 = set(stopwords.words('english'))


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'\"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'emo_pos', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'emo_pos', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', 'emo_pos', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', 'neutral', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'emo_neg', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', 'emo_neg', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = word_tokenize(tweet)
   # words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if word not in stop_words1:
                word = str(lemmatizer.lemmatize(word))
                if word not in stop_words1:
                    processed_tweet.append(word)

    return ' '.join(processed_tweet)


with io.open('positive_tweets.json', 'r', encoding="utf-8") as f:
    #  with io.open('personal5.csv', 'w', encoding="utf-8") as csvf:
    #   linewriter = csv.writer(csvf)
     with io.open('positive_train.txt', 'w') as txtfile:
      #with io.open('personal5.json', 'w') as json_file:
        for line in f:
                tweet1 = json.loads(line)
                tokens = preprocess_tweet(tweet1['text'])
                #print(tokens)
               # linewriter.writerow(tokens)
                txtfile.write(tokens +os.linesep)
                #json.dump(tokens+os.linesep, json_file,indent=6)

               # txt_file.writelines(tokens)
