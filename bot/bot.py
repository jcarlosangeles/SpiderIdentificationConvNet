from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
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
        stream.filter(track=['@arachno_bot'])

class TwitterListener(StreamListener):
    def on_data(self, data):
        try:
            print('Received a tweet..')
            mention = json.loads(data)
            user = mention['user']['screen_name']
            tweet_id = mention['id']
            replies = mention['in_reply_to_status_id']
            retweeted = mention['retweeted']
            classes_eval = list()
            #If mention has media
            if "extended_entities" in mention:
                media = mention['extended_entities']['media']
                num_media = len(media)
                if num_media == 1:
                    if media[0]['type'] == 'photo':
                        url = media[0]['media_url']
                        img = download_img(url)
                        class_eval, result_eval = predict(model, img)
                        classes_eval.append(class_eval)
                        print('Predicted class: %s' % str(class_eval[0]))
                else:
                    for media in mention['extended_entities']['media']:
                        if media['type'] == 'photo':
                            url = media['media_url']
                            img = download_img(url)
                            class_eval, result_eval = predict(model, img)
                            classes_eval.append(class_eval)
                            print('Predicted class: %s' % str(class_eval))

                reply(user, classes_eval, tweet_id);
            else:
            #     print('It has no media on it')
            #     if retweeted == False:
            #         if replies == None:
            #             api.update_status('Hola @' + user + '. Solo puedo identificar arañas si me compartes una foto.', in_reply_to_status_id = tweet_id)
            # print('Replied')
            # print('Keeping streaming...')
                return True
        except BaseException as e:
            print('Error: %s' % str(e))
        return True

    def on_error(self, status):
        print(status)

def reply(user, classes_eval, tweet_id):

    if len(classes_eval) == 1:
        eval_text = '@' + user + ' Parece una araña '
        if classes_eval[0] == 0:
            eval_text += 'de importancia médica. #IM⚠️'
        elif classes_eval[0] == 1:
            eval_text += 'inofensiva. #NIM✔️'
    else:
        eval_text = '@' + user
        for elt in range(len(classes_eval)):
            if classes_eval[elt] == 0:
                curr_eval = ' La imagen ' + str(elt + 1) + ' parece de importancia médica (#IM⚠️). '
            elif classes_eval[elt] == 1:
                curr_eval = ' La imagen ' + str(elt + 1) + ' parece inofensiva (#NIM✔️). '
            eval_text += curr_eval
    eval_text += ' ¿Tú qué opinas, @Arachno_Cosas?'
    print(eval_text)
    api.update_status( eval_text, in_reply_to_status_id = tweet_id)

def download_img(url):
    img_name = 'predict_me_this.jpg'
    urllib.request.urlretrieve(url, img_name)
    return img_name

def predict(model, img):
    test_image = image.load_img(img, target_size = (200, 200))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result_eval = model.predict(test_image)
    class_eval = model.predict_classes(test_image)

    return class_eval, result_eval


if __name__ == "__main__":

    auth = tweepy.OAuthHandler(credentials.consumer_key, credentials.consumer_secret)
    auth.set_access_token(credentials.access_token, credentials.access_token_secret)
    api = tweepy.API(auth)
    model = models.load_model('model_3.h5')
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    print('Bot started. Currently streaming...')
    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets()
