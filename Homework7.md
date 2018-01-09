

```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "Ed4RNulN1lp7AbOooHa9STCoU"
consumer_secret = "P7cUJlmJZq0VaCY0Jg7COliwQqzK0qYEyUF9Y0idx4ujb3ZlW5"
access_token = "839621358724198402-dzdOsx2WWHrSuBwyNUiqSEnTivHozAZ"
access_token_secret = "dCZ80uNRbFDjxdU2EckmNiSckdoATach6Q8zb7YYYE5ER"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Search Term
#BBC, CBS, CNN, Fox, and New York times
target_terms = ("@BBCWorld", "@CBSTweet", "@CNN",
                "@FoxNews", "@nytimes")

# "Real Person" Filters
min_tweets = 5
max_tweets = 100
lang = "en"
num_pages= 5
oldest_tweet = ""

df = pd.DataFrame()

# Array to hold sentiment
sentiments = []

# Variable to hold the list for plotting
rows_list = []
counter = 1

for target_user in target_terms:
    # Variables for holding sentiments
    
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
  
    #lets loop to get the data.
    
    for x in range(num_pages):
        # Get all tweets from home feed
        public_tweets = api.search(target_user , count = (max_tweets / num_pages) ,result_type="recent",max_id=oldest_tweet)
        
        # Loop through all tweets
        for tweet in public_tweets["statuses"]:
            
            # Target String Setting
            target_string = tweet['text']
            target_time= tweet['created_at']
           

            # Run analysis
            compound = analyzer.polarity_scores(target_string)["compound"]
            pos = analyzer.polarity_scores(target_string)["pos"]
            neu = analyzer.polarity_scores(target_string)["neu"]
            neg = analyzer.polarity_scores(target_string)["neg"]
            tweets_ago = counter

            # Add each value to the appropriate list
            compound_list.append(compound)
            positive_list.append(pos)
            neutral_list.append(neu)
            negative_list.append(neg)
            
            # Add sentiments for each tweet into an array
            sentiments.append({"Date": tweet["created_at"],
                          "Compound": compound,
                          "Positive": pos,
                          "Negative": neu,
                          "Neutral": neg,
                          "Tweets Ago": counter,
                          "target_user" : target_user})
       
           # Add to counter
            counter = counter + 1  
    
    #store the averages
    sentiment ={
        "user" : target_user,
    "Compound": np.mean(compound_list),
    "Postive" : np.mean(positive_list),
    "Neutral" :np.mean(neutral_list),
    "Negative" :np.mean(negative_list)
            }

    print(sentiment)
    
# Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)

```

    {'user': '@BBCWorld', 'Compound': -0.040915, 'Postive': 0.047449999999999999, 'Neutral': 0.88880000000000015, 'Negative': 0.063750000000000001}
    {'user': '@CBSTweet', 'Compound': 0.11847142857142859, 'Postive': 0.16, 'Neutral': 0.77128571428571435, 'Negative': 0.068714285714285714}
    {'user': '@CNN', 'Compound': -0.044064999999999993, 'Postive': 0.045149999999999996, 'Neutral': 0.88034999999999997, 'Negative': 0.074550000000000005}
    {'user': '@FoxNews', 'Compound': 0.13750625, 'Postive': 0.083000000000000004, 'Neutral': 0.876, 'Negative': 0.040937500000000002}
    {'user': '@nytimes', 'Compound': -0.039809999999999991, 'Postive': 0.049149999999999999, 'Neutral': 0.82220000000000004, 'Negative': 0.12859999999999999}



```python
# Create data to create the scatter for BBC
bbc_pd = sentiments_pd[sentiments_pd.target_user == '@BBCWorld']
bbc_comp = bbc_pd.mean()
bbc_comp_mean = bbc_comp.loc["Compound"]
print (bbc_comp_mean)
```

    -0.040915



```python
area = 100
plt.scatter(x = bbc_pd["Tweets Ago"] , y =bbc_pd["Compound"], s=area, c="g", alpha=0.5,edgecolors="black" ,label = '@BBC' )

# Incorporate the other graph properties
plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")
plt.grid(True)

# Show plot
plt.show()

# Save the figure
plt.savefig("BBCWORLD.png")
```


![png](Homework7_files/Homework7_2_0.png)



```python
# Build a scatter plot for CBS 

CBSTweet_pd = sentiments_pd[sentiments_pd.target_user == '@CBSTweet']
CBS_comp = CBSTweet_pd.mean()
CBS_comp_mean = CBS_comp.loc["Compound"]
print (CBS_comp_mean)

#Scatter for CBS
plt.scatter(x = CBSTweet_pd["Tweets Ago"] , y =CBSTweet_pd["Compound"], s=area, c="r", alpha=0.5,edgecolors="black" ,label = '@CBS' )


# Incorporate the other graph properties
plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")
plt.grid(True)

# Show plot
plt.show()

# Save the figure
plt.savefig("CBS.png")



```

    0.118471428571



![png](Homework7_files/Homework7_3_1.png)



```python
# Build a scatter plot for Fox News

FoxNews_pd = sentiments_pd[sentiments_pd.target_user == '@FoxNews']
FoxNews_comp = FoxNews_pd.mean()
FoxNews_comp_mean = FoxNews_comp.loc["Compound"]
print (FoxNews_comp_mean)
#print(FoxNews_pd.head)

# Incorporate the other graph properties
plt.scatter(x = FoxNews_pd["Tweets Ago"] , y =FoxNews_pd["Compound"], s=area, c="b", alpha=0.5,edgecolors="black" ,label = '@FoxNews' )

plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")
plt.grid(True)


# Show plot
plt.show()
plt.grid(True)

# Save the figure
plt.savefig("FoxNews.png")

```

    0.13750625



![png](Homework7_files/Homework7_4_1.png)



```python
# Build a scatter plot for CNN

CNN_pd = sentiments_pd[sentiments_pd.target_user == '@CNN']
CNN_comp = CNN_pd.mean()
CNN_comp_mean = CNN_comp.loc["Compound"]
print (CNN_comp_mean)

plt.scatter(x = CNN_pd["Tweets Ago"] , y =CNN_pd["Compound"], s=area, c="y", alpha=0.5,edgecolors="black" ,label = '@CNN' )

#print(FoxNews_pd.head)

# Incorporate the other graph properties
plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")
plt.grid(True)

# Show plot
plt.show()

# Save the figure
plt.savefig("CNN.png")            
```

    -0.044065



![png](Homework7_files/Homework7_5_1.png)



```python
# Build a scatter plot for NY Times

nytimes_pd = sentiments_pd[sentiments_pd.target_user == '@nytimes']
nytimes_comp = nytimes_pd.mean()
nytimes_comp_mean = nytimes_comp.loc["Compound"]
print (nytimes_comp_mean)


plt.scatter(x = nytimes_pd["Tweets Ago"] , y =nytimes_pd["Compound"], s=area, c="red", alpha=0.5,edgecolors="black" ,label = '@NYTIMES' )

#print(FoxNews_pd.head)

# Incorporate the other graph properties
plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")

# Show plot
plt.show()
plt.grid(True)


# Save the figure
plt.savefig("nytimes.png")

```

    -0.03981



![png](Homework7_files/Homework7_6_1.png)



```python
area = 100
plt.scatter(x = bbc_pd["Tweets Ago"] , y =bbc_pd["Compound"], s=area, c="g", alpha=0.5,edgecolors="black" ,label = '@BBC' )
plt.scatter(x = CBSTweet_pd["Tweets Ago"] , y =CBSTweet_pd["Compound"], s=area, c="r" , alpha=0.5,edgecolors="black" ,label = '@CBS' )
plt.scatter(x = FoxNews_pd["Tweets Ago"] , y =FoxNews_pd["Compound"], s=area, c="b", alpha=0.5,edgecolors="black" ,label = '@FoxNews' )
plt.scatter(x = CNN_pd["Tweets Ago"] , y =CNN_pd["Compound"], s=area, c="y", alpha=0.5,edgecolors="black" ,label = '@CNN' )
plt.scatter(x = nytimes_pd["Tweets Ago"] , y =nytimes_pd["Compound"], s=area, c="red", alpha=0.5,edgecolors="black" ,label = '@NY Times' )


plt.legend(target_terms,scatterpoints=1,loc='best',ncol=3,fontsize=8)
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.title("Sentiment Analysis of Tweets")

# Incorporate the other graph properties
plt.title("Sentimental Tweet analysis")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets")

# Show plot
plt.show()
plt.grid(True)
```


![png](Homework7_files/Homework7_7_0.png)



```python
# Create an array that contains the number of users each language has
users = target_terms
x_axis = np.arange(len(users))

# Sets the x limits of the current chart
plt.xlim(-1, len(x_axis))

plot_values = [bbc_comp_mean,CBS_comp_mean,CNN_comp_mean,FoxNews_comp_mean,nytimes_comp_mean]

plt.ylim(-0.50,max(plot_values)+0.25)
         
# Tell matplotlib where we would like to place each of our x axis headers
tick_locations = [value+2.5 for value in x_axis]
plt.xticks(tick_locations, [target_terms])

my_colors = 'rgbkymc'

plt.bar(x_axis, plot_values,color= my_colors, alpha=0.5, align="edge")
plt.show()
plt.grid(True)

```


![png](Homework7_files/Homework7_8_0.png)

