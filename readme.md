# GCP Pipeline Instructions

## Data Collection / Feature Engineering
The first part of your pipeline should have a notebook pulling in relevant Southwest Twitter data through Twitter’s official API. On a low-cost budget, we recommend generating several API keys and rotating pulls between notebooks. This will only be a factor for twitter_ingestion_auth notebooks. Below are some notes on how to run the notebooks.

### Getting Tweet Data From Twitter 
Notebook: twitter_ingestion_auth
Runtime: A couple minutes
Frequency: Hourly, Use Several API keys if possible

Description: Pull tweet data from Twitters official API, including tweet identifier, time created, text, who wrote it, etc.

### Getting User Data From Twitter
Notebook: pull_users
Runtime: A couple minutes
Frequency: Once, daily

Description: Everytime you refresh your twitter ingestion, this notebooks checks if we have any new users. If so, we pull their user information like name, description, handle.

Note: Current implementation only pulls new users, not updating previous users, but can be accomplished with additional keys, funding etc.

### Aggregating and Cleaning Twitter Data
Notebook: aggregated_twitter_pulls
Runtime: A couple of minutes
Frequency: Hourly

This script takes all the raw Twitter pulls and compiles them into our core dataset. We also include user attributes for convenience. The output tables live in ‘processed’, and will be our main table moving forward throughout the analysis. 
Topic Modeling
Keep in mind that topic modeling currently gives us qualitative insights to decide how to run the rest of the analysis. There is an opportunity to dynamically monitor the relevant topics of the day for the rest of the analysis, but the manual filters currently being used should suffice

Notebook: Southwest_top2vec
Runtime: A couple minutes
Frequency: (Optional)

## Tweet Similarity
Now that we have clean Twitter data to analyze, we are going to run the first step in our machine learning workflow. The following notebooks have a lot more requirements than the previous step. All should run the latest version of pytorch as a boot image (Docker is pytorch/pytorch) and should be run with at least 26GB of RAM. 

These following steps are by the far most intensive notebooks to run in the series. When possible we will use NVIDIA GPUs to dramatically accelerate the runtime speed. There are likely opportunities to optimize the runtime, but in the meantime we recommend to run as is.

One of the important aspects of this analysis is that we store the results in objects, not Dataframes. This helps reduce space and complexity. Rather than outputting a gigantic table listing all the relationships, we are going to just record the relationship within each Tweet Object. Then, when the analysis is done we export the objects, as a giant table would not be feasible.

### BERT Similarity Matrix
This notebooks takes our filtere twitter dataset (the core dataset filtered on keywords illuminated by topic modeling) and takes each text tweet and compares it with each other. The output is a massive square matrix with the similarity scores. As we capture more tweets, the time increase quadratically, since our matrix is NxN. However, we can use an NVIDIA T4 GPU to accelerate performance from several hours to several minutes.

Notebook: 
Runtime: 
Frequency: 
Specs: At least 4 VCPUs, 26GB RAM



### Tweet Similarity Scoring
Notebook: similarity_matrix-auto
Runtime: Almost half a day
Frequency: Weekly
Specs: 4 vCPUs 26 GB RAM

Notes: By far one of the longer scripts. Opportunities to optimize
Output: twt2twt_w_score.pkl, auth2auth_w_score.pkl

Aspect Based Sentiment Analysis 
This script takes the tweets filtered, and creates aspect and sentiment tags for each tweet. It then stores those tags to the Tweet objects (twt2twt_w_score.pkl) we generated in the previous step. Like above, we can accelerate a several hour analysis by running it through the same GPU. Original runtime is 5+ hours, but GPU acceleration can bring the runtime to around 90 minutes. 

Notebook: aspect_based_sentiment_auto
(Rest same as above)
Runtime: ~90 minutes
Output: twt2twt_w_score_w_sentiments.pkl

### Relationships to Dataframe
This is the final step of our Text analysis pipeline. Here we take the relationships in the objects and convert them into a table, so that we can populate the relationships in our network graph.

Notebook: relationship_to_df
(Rest same as above)
Runtime: ~8 hours
Output: tweet2tweet_df3.pickle, author2author_df3.pickle

### Sentiments to Dataframe
This notebook just takes the output of the table above and integrates the sentiment results for Neo4J consumption.
Notebook: sentiments_to_df
(Rest same as above)
Runtime: A couple minutes
Output: tweet2tweet_df4
Note: Before running in Neo4J, please run this code to make sure the data is clean before ingesting


dftweet = pd.read_pickle("tweet2tweet_df4.pickle")
dftweet = dftweet.fillna('N/A')
dftweet.to_csv("TweetToTweetData.csv")





