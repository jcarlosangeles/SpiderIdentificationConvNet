# from tensorflow.keras import models
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras import optimizers
from tweepy.streaming import StreamListener
import json
import numpy as np
import urllib.request
import tweepy
import credentials

class TwitterStreamer():
    #Class for streaming and processing live tweets
    def stream_tweets(self):
        listener = TwitterListener()
        stream = tweepy.Stream(auth, listener)
        #stream.filter(follow=["1047966576236847104"]) #Arachno_Cosas
        stream.filter(follow=["1180716992"]) #Yo

class TwitterListener(StreamListener):
    #Regular listener class that just prints data
    def on_data(self, data):
        try:
            tweet = json.loads(data)
            text = tweet['text']
            retweeted = tweet['retweeted']
            replies = tweet['in_reply_to_status_id']
            media = tweet['']
            if retweeted != None and replies != None:
                if '#IM⚠️' in text:
                    print('Tiene foto peligrososa')
                    if "extended_entities" in mention:
                        for media in mention['extended_entities']['media']:
                            if media['type'] == 'photo':
                                #Download image and save it as dangerous
                                url = media['media_url']
                                img = download_img(url)


            return True
        except BaseException as e:
            print('Error: %s' % str(e))
        return True
    def on_error(self, status):
        print(status)

def download_img(url):
    img_name = 'predict_me_this.jpg'
    urllib.request.urlretrieve(url, img_name)
    return img_name


if __name__ == "__main__":

    auth = tweepy.OAuthHandler(credentials.consumer_key, credentials.consumer_secret)
    auth.set_access_token(credentials.access_token, credentials.access_token_secret)
    api = tweepy.API(auth)

    print('Bot starting')
    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets()
