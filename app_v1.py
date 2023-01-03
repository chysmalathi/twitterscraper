# -*- coding: utf-8 -*-

#################### Basic Settings
import streamlit as st

st.set_page_config(page_title='Twitter Anaysis',page_icon='happy sad.jpg', layout="wide") #, page_icon='atom.png', initial_sidebar_state="auto", menu_items=None)


#################### Aesthetics
hide_st_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

########

c1,c2,c3 = st.columns([2,4,2], gap='large')

c2.markdown("<h1 style='text-align: center;'><font face='High Tower Text'> Twitter Analysis </font></h1>", unsafe_allow_html=True)

st.markdown("***", unsafe_allow_html=True)

#################################################################################################################### Importing Dependencies

from snscrape.modules import twitter as tw
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en_core_web_sm')
from datetime import datetime as dt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords= list(set(nltk.corpus.stopwords.words('english')))
import re
import pickle
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import urllib3, socket
from urllib3.connection import HTTPConnection

############################# The search Query

############### WORDS

st.markdown("<h5 style='text-align: left;'>Words</h5>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1,3,3,3,1], gap='large')

# all_of_these_words = ''
all_of_these_words = c2.text_input( 'All of these words' , value='')#'nft' #

# this_exact_phrase = ""
this_exact_phrase = c2.text_input( 'This exact Phrase' , value='')
this_exact_phrase = f'"{this_exact_phrase}"' if this_exact_phrase != '' else '' #

# any_of_these_words = ""
any_of_these_words = c3.text_input( 'Any of these words  (Words separated with spaces)' , value='') #'any of these words'# 
any_of_these_words = "(" + any_of_these_words.replace(" ", " OR ") + ")" if any_of_these_words  != '' else ''

none_of_these_words = ""
none_of_these_words = c4.text_input( 'None of these words  (Words separated with spaces)' ,  value='') #'none of these words' #
none_of_these_words = "-" + none_of_these_words.replace(" ", " -") if none_of_these_words != '' else ''

# these_hashtags = ""
these_hashtags = c3.text_input( 'These hashtags (Words separated with spaces)' , value='') #'' #
these_hashtags = "#" + these_hashtags.replace(" ", " OR #") if these_hashtags != '' else ''
these_hashtags = "(" + these_hashtags.replace("##", "#") + ")" if these_hashtags != '' else '' 

############### ACCOUNTS

st.markdown("<h5 style='text-align: left;'>Accounts</h5>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1,3,3,3,1], gap='large')

# from_any_of_these_accounts = ""
from_any_of_these_accounts = c3.text_input( 'from_any_of_these_accounts  (User ids separated with spaces)' , value='') #''#
from_any_of_these_accounts = "(from:" + from_any_of_these_accounts.replace(" ", " from:") + ")" if from_any_of_these_accounts != '' else ''

# to_any_of_these_accounts = ""
to_any_of_these_accounts = c4.text_input('To any of these accounts  (User ids separated with spaces)', value='') #
to_any_of_these_accounts = "(to:" + to_any_of_these_accounts.replace(" ", " to:") + ")" if to_any_of_these_accounts != '' else ''

# mentioning_any_of_these_accounts = ""
mentioning_any_of_these_accounts = c2.text_input('Mentioning any of these accounts  (User ids separated with spaces)', value='') #
mentioning_any_of_these_accounts = "@" + mentioning_any_of_these_accounts.replace(" ", " @") if mentioning_any_of_these_accounts != '' else ''
mentioning_any_of_these_accounts = "(" + mentioning_any_of_these_accounts.replace("@@", "@") + ")" if mentioning_any_of_these_accounts != '' else ''

############### ENGAGEMENT

st.markdown("<h5 style='text-align: left;'>Engagement</h5>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1,3,3,3,1], gap='large')

# post_containing_min_likes = ""
post_containing_min_likes = c3.number_input('Minimum Likes', step=1) #
post_containing_min_likes = 'min_faves:' + str(post_containing_min_likes) if post_containing_min_likes != 0 else ''

# post_containing_min_replies = ""
post_containing_min_replies = c4.number_input( 'Minimum Replies' , step=1) #
post_containing_min_replies = "min_replies:" + str(post_containing_min_replies) if post_containing_min_replies  != 0 else ''

# post_containing_min_retweets = ""
post_containing_min_retweets = c2.number_input('Minimum Retweets' , step=1) #2000 #
post_containing_min_retweets = 'min_retweets:' + str(post_containing_min_retweets) if post_containing_min_retweets != 0 else ''

############### DATES

st.markdown("<h5 style='text-align: left;'>Dates</h5>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1,3,3,3,1], gap='large')

if c2.selectbox('Search Duration', ['All Time', 'Between Dates']) == 'All Time':
    posts_from_date = ''
    posts_till_date = ''
else:
    
    # posts_from_date = ""
    posts_from_date = c3.date_input('From Date' , value=None) #'2022-10-01' #
    posts_from_date = dt.strftime(posts_from_date, "%Y-%m-%d")
    posts_from_date = "since:" + posts_from_date if posts_from_date != '' else ''
    
    # posts_till_date = ''
    posts_till_date = c4.date_input('To Date') #'2022-11-23' #
    posts_till_date = dt.strftime(posts_till_date, "%Y-%m-%d")
    posts_till_date = "until:" + posts_till_date if posts_till_date != '' else ''


st.markdown("<h5 style='text-align: left;'>Max Results</h5>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1,3,3,3,1], gap='large')
max_tweets = c2.number_input('Max Tweets(Enter 0 for scraping all the tweets available)', value=100)

query = ''
for i in [all_of_these_words , this_exact_phrase , any_of_these_words , none_of_these_words , these_hashtags  , 'lang:en' , from_any_of_these_accounts , to_any_of_these_accounts , 
          mentioning_any_of_these_accounts , post_containing_min_replies , post_containing_min_likes , post_containing_min_retweets , posts_till_date , posts_from_date]:
    if i != '':
        query += i + " "
query = query if query == all_of_these_words + 'lang:en' else query[:-1]

@st.cache(allow_output_mutation=True)
def collect_tweets(query):
    df = pd.DataFrame(columns=['Link', 'CreatedUTCTimestamp', 'Author', 'AuthorsFollowers', 'Text', 'Likes', 'RetweetCount', 'ReplyCount', 'QuoteCount', 'Type' , 'ParentTweetId', 'Hashtags'])

    for i,v in enumerate(tw.TwitterSearchScraper(query).get_items()):
        x = json.loads(v.json())
        df = df.append(pd.DataFrame([ x['url'], dt.strptime(x['date'], "%Y-%m-%dT%H:%M:%S+00:00"), x['user']['username'], x['user']['followersCount'], x['content'], x['likeCount'], x['retweetCount'] , x['replyCount'], x['quoteCount'], 'Reply' if x['inReplyToTweetId'] != None else 'Retweet' if x['retweetedTweet'] != None else 'Quote Tweet' if  x['quotedTweet'] != None else 'Tweet'   ,  x['inReplyToTweetId'] if x['inReplyToTweetId'] != None else x['quotedTweet']['id'] if x['quotedTweet'] != None else 'NA' ,  x['hashtags'] if x['hashtags'] != None else []  ] ,  index=['Link', 'CreatedUTCTimestamp', 'Author', 'AuthorsFollowers', 'Text', 'Likes', 'RetweetCount', 'ReplyCount', 'QuoteCount' , 'Type' , 'ParentTweetId', 'Hashtags']).T, ignore_index=True)
        
        
        if max_tweets != 0:
            if i == max_tweets-1:
                break
    return df


if st.checkbox('Search Tweets'):

    with st.spinner('Please wait while we collect your data ⌛'):
        df = collect_tweets(query)    
        st.markdown(f'Collected {df.shape[0]} Rows of Data')
        st.dataframe(df)
    
    def basic_cleaning(text):
        text = re.sub(r'^rt ' , "" , text)
        text = re.sub('  ' , "" , text)
        text = re.sub( "(@\S*)" ," ",text) #mentions removal
        text = re.sub( "#\S*" ," ",text)   #aashtags removal
        text = re.sub( "\(\)" ," ",text) 
        text = re.sub( r"http\S*" ," ",text) #Links removal
        text = text.replace("\\n"," ")
        text = re.sub(r"[/.\';:,-]" , " " , text)
        text = re.sub(r'[\/*:;"-+=“”‘’.,]' , " " , text) 
        text = text.strip()
        return text
    df["ProcessedText"] = df["Text"].apply(basic_cleaning)
    
    
    ################################################################################################################################################# Remove Stopwords
    
    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in stopwords])
    
    # df["ProcessedText"] = df["ProcessedText"].apply(lambda text: remove_stopwords(text))
    
    ################### Removing Specific Words
    
    remove_words = st.checkbox('Remove specific words?')
    
    if remove_words:
        c1,c2,c3 = st.columns([1,3,7], gap='large')
        words_to_remove = c2.text_input('Enter the words to remove separated by commas "," (Case Sensitive)')
        
        words_list = [word.strip() for word in words_to_remove.split(',')]
        
        def remove_specific_text(text):
            for word in words_list:
                text = re.sub(word.title(), '', text)
                text = re.sub(word.upper(), '', text)
                text = re.sub(word.lower(), '', text)
                text = re.sub(word, '', text)
            return re.sub('\s+' , ' ' ,text)
        
        df['ProcessedText'] = df['ProcessedText'].apply(remove_specific_text)
        
    # Visualizing the words
    def wordcloudviz(data, column):
        fig, ax = plt.subplots()
        all_for_wc = " ".join([ i for i in data[column].apply(remove_stopwords)])
        wordsviz = WordCloud(width = 1000, height = 500 , max_font_size=100).generate(all_for_wc)
        ax.imshow(wordsviz , interpolation='bilinear')
        ax.axis('off')
        c1, c2, c3 = st.columns([2,6,2])
        c2.pyplot(fig)
    st.markdown('***', unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left;'>WordCloud</h5>", unsafe_allow_html=True)
    
    with st.spinner('Loading Wordcloud...'):
        wordcloudviz(df,'ProcessedText')
    
    c1,c2,c3,c4,c5 = st.columns([1,2,1,2,1], gap='large')
    
    # if c1.checkbox('Get Sentiment?'): # Model Updation Feature if needed!
    #     with st.spinner('Updating the Model...⌛'):          
    #         try:
    #             # Try to get model online
    #             MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    #             tokenizer = AutoTokenizer .from_pretrained(MODEL)
    #             model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    #             pickle.dump(MODEL, open('Roberta_Model.sav', 'wb'))
    #         except:
    #             st.write('Unable to get the model.')

    st.markdown('***', unsafe_allow_html=True)

    menulist=["Unigram","Bigram","Trigram"] 
    choice = st.selectbox("Select Graphs", menulist)
    if choice=="Unigram":
        st.subheader("Unigram")
        with st.spinner('Getting the Unigram in place... ⌛'):
            def lemmatizer(text):
                sent = []
                doc = nlp(text)
                for word in doc:
                    sent.append(word.lemma_)
                return " ".join(sent)
            df["review_lemmatize"] =  df.apply(lambda x: lemmatizer(x['ProcessedText']), axis=1)
            def get_top_n_bigram(corpus, n=None):
                vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]
            common_words = get_top_n_bigram(df['review_lemmatize'], 30)

            df3 = pd.DataFrame(common_words, columns = ['unigram' , 'count'])
            fig = go.Figure([go.Line(x=df3['unigram'], y=df3['count'])])
            fig.update_layout(title=go.layout.Title(text="Top 30 unigram in the question text after removing stop words and lemmatization"))
        with st.spinner('Loading Unigram...'):
            st.plotly_chart(fig, use_container_width=True)
    elif choice == "Bigram":
        st.subheader("Bigram")  
        with st.spinner('Getting the Bigram in place... ⌛'):
            def lemmatizer(text):
                sent = []
                doc = nlp(text)
                for word in doc:
                    sent.append(word.lemma_)
                return " ".join(sent)
            df["review_lemmatize"] =  df.apply(lambda x: lemmatizer(x['ProcessedText']), axis=1)
            def get_top_n_bigram(corpus, n=None):
                vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]
            common_words = get_top_n_bigram(df['review_lemmatize'], 30)

            df3 = pd.DataFrame(common_words, columns = ['bigram' , 'count'])
            fig = go.Figure([go.Bar(x=df3['bigram'], y=df3['count'])])
            fig.update_layout(title=go.layout.Title(text="Top 30 bigrams in the question text after removing stop words and lemmatization"))
        with st.spinner('Loading Bigram...'):
            st.plotly_chart(fig, use_container_width=True)
        # fig.show()
    elif  choice == "Trigram":
        st.subheader("Trigram")  

    if c1.checkbox('Get Sentiment?'):
        with st.spinner('Getting the model in place... ⌛'):
            def model():
                st.write("in model")
                MODEL = pickle.load(open('Roberta_Model.sav', 'rb'))
                st.write("fileopend")
                tokenizer = AutoTokenizer.from_pretrained(MODEL)
                st.write("token")        
                model = AutoModelForSequenceClassification.from_pretrained(MODEL)
                st.write("automodel")
                return model, tokenizer
            
            model, tokenizer = model()
        
        with st.spinner('Getting the Sentiments DataFrame... ⌛'):
            scores_list = []
            negative_score = []
            neutral_score = []
            positive_score = []
        
            for i in df['ProcessedText']:
                if i != '':
                    encoded_text = tokenizer(i, return_tensors='pt')
                    output = model(**encoded_text)
                    st.write("in if")
                    scores = softmax(output[0][0].detach().numpy())
                    scores_list.append(0 if scores[0] == scores[2] else -scores[0] if scores[0] == scores.max() else 0 if scores[1] == scores.max() else scores[2])
                    negative_score.append(float(str(scores[0])[:6]))
                    neutral_score.append(float(str(scores[1])[:6]))
                    positive_score.append(float(str(scores[2])[:6]))
                    # scores_list.append(scores)
                else:
                    scores_list.append(0)
        
            df['OverallSentimentScore'] = scores_list
            df['NegativeSentimentScore'] = negative_score
            df['NeutralSentimentScore'] = neutral_score
            df['PositiveSentimentScore'] = positive_score
            df['Title_Remark'] = df['OverallSentimentScore'].apply(lambda x: 'Positive' if x >= 0.05 else 'Negative' if x < -0.05 else 'Neutral')
            df = df.iloc[:,[14,15,16,17,18,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
        
        st.markdown('***', unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: left;'>Dataframe with Sentiment Scores</h5>", unsafe_allow_html=True)
        st.dataframe(df)

        
        st.markdown('***', unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: left;'>Sentiments Counts</h5>", unsafe_allow_html=True)
        with st.spinner('Getting the sentiment counts.. ⌛'):
            # Graph (Pie Chart in Sidebar)
            df_target = df[['Link', 'Title_Remark']].groupby('Title_Remark').count() / len(df)
            fig_target = go.Figure(data=[go.Pie(labels=df_target.index,
                                                values=df_target['Link'],
                                                hole=.3)])
            fig_target.update_layout(showlegend=False,
                                     height=200,
                                     margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
            fig_target.update_traces(textposition='inside', textinfo='label+percent')

            # Layout (Sidebar)
            # cat_selected = st.selectbox('Categorical Variables', vars_cat)
            # cont_selected = st.selectbox('Continuous Variables', vars_cont)
            # cont_multi_selected = st.multiselect('Correlation Matrix', vars_cont,
            #                                      default=vars_cont)
            st.plotly_chart(fig_target, use_container_width=True)
        
        st.markdown('***', unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: left;'>Sentiments Trend</h5>", unsafe_allow_html=True)
        with st.spinner('Getting you the Time-Series plot.. ⌛'):
            fig = px.line(df.sort_values(by='CreatedUTCTimestamp'), 'CreatedUTCTimestamp', 'OverallSentimentScore')
            # fig.update_traces(lin)
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                        ])))
            fig.update_traces(line_color='red' if np.mean(df['OverallSentimentScore']) < 0.05 else 'green' if np.mean(df['OverallSentimentScore']) > 0.05 else 'blue')
            fig.update_layout( xaxis_title="Created Timestamp", yaxis_title="Overall Sentiment Score", height=800, yaxis_range=[-1,1])
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('***', unsafe_allow_html=True)
            
        st.markdown("***", unsafe_allow_html=True)
            
    if st.checkbox('Continue with Topic Modelling?'):
        with st.spinner('Vectorizing the Text...⌛'):
            vect =TfidfVectorizer(stop_words=stopwords,max_features=1000)
            vect_text=vect.fit_transform(df['ProcessedText'])
            
        with st.spinner('Building the LDA Model...⌛'):
            from sklearn.decomposition import LatentDirichletAllocation

            c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2], gap='large')
            
            num_topics = c2.number_input('Enter number of topics to be extracted from the corpus.', value=7, step=1)

            lda_model=LatentDirichletAllocation(n_components=num_topics, learning_method='online',random_state=25) 
            lda_top=lda_model.fit_transform(vect_text)

        with st.spinner('Generating the Word Clouds...⌛'):
            import matplotlib.colors as mcolors
            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
            from wordcloud import STOPWORDS
            stopwords = list(set(STOPWORDS))
            import math
            figsize=math.ceil(lda_model.n_components/3)

            fig, axes = plt.subplots(figsize, 3, figsize=(30,7*figsize))#, sharex=True, sharey=True)

            from itertools import zip_longest

            #'family': 'serif',
            font = {
                    'color':  'black',
                    'weight': 'bold',
                    'size': 25,
                    }
            cloud = WordCloud(width = 250,height = 250, background_color ='white', stopwords = stopwords, colormap='tab10', color_func=lambda *args, **kwargs: cols[i], max_font_size=60)

            
            vocab = vect.get_feature_names_out()
            for i, comp, ax in zip_longest(range(lda_model.n_components), lda_model.components_ , axes.flatten()):
                try:
                    vocab_comp = zip(vocab, comp)
                    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:20]
                    fig.add_subplot(ax)
                    cloud.generate_from_text(' '.join(dict(sorted_words).keys()))
                    plt.gca().set_title('Topic ' + str(i), fontdict=font)
                    plt.gca().imshow(cloud)
                except:
                    pass
                ax.axis('off')
            plt.subplots_adjust(wspace=200, hspace=50)
            plt.axis('off')
            plt.margins(x=500, y=500)
            plt.tight_layout()
            st.markdown("Topic wise WordClouds")
            c1,c2,c3 = st.columns([1,10,1])
            c2.pyplot(fig)

        st.markdown('***', unsafe_allow_html=True)
        
        
