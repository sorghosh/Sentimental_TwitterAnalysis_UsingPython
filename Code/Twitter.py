from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import Prediction


ckey = "xxxxxxx"
csecret = "xxxxxxx"
atoken = "xxxxxxx"
asecret = "xxxxxxx"

class listener(StreamListener):
    def on_data(self,data):
        all_data = json.loads(data)
        if all_data["lang"] == "en":
            text = all_data["text"]
            geo  = all_data["geo"]
            place = all_data["place"]
            word = Prediction.get_features(text)
            print "tweeter text >>>>" ,text
            print "location>>>",geo
            print "place>>>",place
            print "sentiments>>>>>",Prediction.voterclassify(word)
        return True
    def on_error(self,status):
        print status

auth = OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)
twitterStream = Stream(auth,listener())
twitterStream.filter(track = ['@British_Airways'])

