# Importing all required libraries.
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob import Word
import string
import re
from IPython.display import display # Allows the use of display() for DataFrames
import warnings
warnings.filterwarnings('ignore')
import subprocess
cmd=['python3','-m','textblob.download_corpora']
subprocess.run(cmd)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css('style.css')
# Adding title of project.
st.markdown("<h1 style='text-align: center; color: red;'>Amazon Reviews Genuinity</h1>", unsafe_allow_html=True)
#st.title('Amazon Reviews Genuinity')

# st. cache is used to store cache data.
# It is used to reduce the loading time of data.
@st.cache(allow_output_mutation=True)

# load_data function is for creating a data frame and deleting the duplicates.
def load_data(nrows):
    data=pd.read_json('new_Electronics_5.json',lines=True,orient='columns',nrows=nrows)
    data=data.drop_duplicates(subset=["reviewText"], keep='first', inplace=False)
    return data

st.write('')
st.write('')
st.write('')

# Now let's load data.
# Here we are loading only 5000 records frow around 1687169.
# We are taking only 5000 since it requires less time to load and we are using them just to show an example.

#data_load_state = st.text('Loading data...')
data = load_data(2000)
#data_load_state.text("Data Loaded !")

# Below code will create a checkbox which when selected will show us raw data.
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write('Total no of reviews are : %s' %len(data.index))
    st.dataframe(data)

# Below code is used to give extra lines space. It is used just for a clean presentation.
st.write('')
st.write('')
st.write('')

# Now let's see reviews based on score ratings.
st.markdown("<h2 style='text-align: center; color: #e056fd;'>See Reviews based on Score Ratings:</h2>", unsafe_allow_html=True)
review_score_filter = st.slider('', 1, 5, 3)
filtered_data = data[data['overall'] == review_score_filter]
st.dataframe(filtered_data)

st.write('')
st.write('')
st.write('')

# Below code is used to show histogram.
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h2 style='text-align: center; color: #e056fd;'>Division of Reviews based on Score:</h2>", unsafe_allow_html=True)
hist_values = data.overall.hist(bins=5,grid=False)
st.pyplot(figure='hist_values')

st.write('')
st.write('')
st.write('')

# Here we are taking only 30 reviews for classification into genuine and fake reviews.

df2 = load_data(2000)
df2=df2.iloc[:, [5,4]]

# Below function classifies ratings into Positive, Negative, and Neutral.
def score_classify(x):
    if x>3:
        return 'Positive'
    elif x<3:
        return 'Negative'
    else:
        return 'Neutral'
df2['Overall_Sentiment']=df2.apply(lambda x: score_classify(x['overall']),axis=1)

df2.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)

# Below code is used to classify reviews based on the sentiment of review text.
sentiment_score_list = []
sentiment_label_list = []

for i in df2['reviewText'].values.tolist():
    sentiment_text=TextBlob(i)
    sentiment_score = sentiment_text.sentiment.polarity

    if sentiment_score > 0:
        sentiment_score_list.append(sentiment_score)
        sentiment_label_list.append('Positive')
    elif sentiment_score < 0:
        sentiment_score_list.append(sentiment_score)
        sentiment_label_list.append('Negative')
    else:
        sentiment_score_list.append(sentiment_score)
        sentiment_label_list.append('Neutral')
    
df2['Review_Sentiment'] = sentiment_label_list
df2['sentiment score'] = sentiment_score_list

# Now we are going to compare the two sentiments- based on ratings and based on text. And write if the review is genuine(True) or fake (False).
comparison_column = np.where(df2["Overall_Sentiment"] == df2["Review_Sentiment"], True, False)
df2["result"] = comparison_column

# Now we are going to remove all Neutral reviews.
df2 = df2[df2.Overall_Sentiment != 'Neutral']
df2 = df2[df2.Review_Sentiment != 'Neutral']

# We are now dividing reviews into two data frames, one which contains a genuine review and one which contains fake.
df3 = df2[df2.result == True]
df4= df2[df2.result != True]

# Let's display the two data frames using the radio button.
st.markdown("<h2 style='text-align: center; color: #e056fd;'>Select option to see Genuine and Fake reviews:</h2>", unsafe_allow_html=True)
DataType=st.radio('',('Genuine','Not Genuine'))
if DataType=='Genuine':
    st.dataframe(df3,width=2500,height=500)
elif DataType=='Not Genuine':
    st.dataframe(df4,width=2500,height=500)


st.write('')
st.write('')
st.write('')

# Preprocess function to be used later.
def preprocess(x):
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")\
                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
                           .replace("'cause'"," because")
    
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x=re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',' ',x)
    x=re.sub(r"\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',x)
    x=re.sub(r'<.*?>',' ',x)
    x=re.sub('[^a-zA-Z]',' ',x)
    x=''.join([i for i in x if not i.isdigit()])
    x=" ".join([Word(word).lemmatize() for word in x.split()])
    return x

# Creating a UI for the user to test if their review is genuine or not.
st.markdown("<h1 style='text-align: center; color: red;'><b>Test Genuinity of Your Review Here</b></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #e056fd;'>Enter Your Product Review</h3>", unsafe_allow_html=True)
user_review = st.text_area("")
st.markdown("<h3 style='text-align: center; color: #e056fd;'>Enter Your Product Ratings :</h3>", unsafe_allow_html=True)
user_score = st.selectbox("",[1,2,3,4,5],index=2)

user_score_senti=score_classify(user_score)
user_review = user_review.lower()
user_review = user_review.replace('[^\w\s]','')
user_review= preprocess(user_review)

def analyze_sentiments(cleaned_verified_reviews):
    analysis=TextBlob(cleaned_verified_reviews)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

Review_Senti=analyze_sentiments(user_review)

def check():
    if st.button('Submit Review'):
        st.write('')
        st.write('')
        if (user_review )=='':
            st.error('Please Enter some review')
            return
        elif (Review_Senti=='Neutral' or user_score_senti=='Neutral'):
            st.success('Review Submitted! Genuine')
        elif (Review_Senti==user_score_senti):
            st.success('Review Submitted! Genuine')
        elif (Review_Senti!= user_score_senti):
            st.warning('Review Submitted! Not Genuine. Please recheck your review.')

check()
