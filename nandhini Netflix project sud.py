#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Nflixdata = pd.read_csv('/Users/nandhiniravi/Desktop/netflix_titles.csv')


# In[3]:


Nflixdata.head()


# In[4]:


Nflixdata['country'] = Nflixdata['country'].fillna(Nflixdata['country'].mode()[0]) #filling the empty values with the frequently repeated country to prevent loss of data
Nflixdata['cast'].replace(np.nan, 'No Data',inplace  = True) # replacing na values with no dataa
Nflixdata['director'].replace(np.nan, 'No Data',inplace  = True)


# In[5]:


Nflixdata.dropna(inplace=True)


# In[6]:


Nflixdata.drop_duplicates(inplace= True) # dropping duplicates


# In[7]:


Nflixdata.shape


# In[8]:


Nflixdata.count()


# In[9]:


Nflixdata.info()


# In[10]:


Nflixdata.drop(labels= 'show_id' ,  axis = 1 , inplace = True )
Nflixdata.head()


# In[11]:


df1= pd.read_csv('/Users/nandhiniravi/Desktop/IMDb movies.csv')
df1 = df1[df1['avg_vote'].notna()]

df2=df1[['title','genre','avg_vote']]
df2["avg_vote"] = 10 * df2["avg_vote"]
df2


# In[12]:


df3=pd.merge(Nflixdata,df2,on="title",how="inner") #merging two dataframe by giving an inner join
df3 


# In[13]:


df3=df3.drop(['listed_in'], axis = 1)


# In[14]:


df3['type'].value_counts()


# In[15]:


df4 = pd.DataFrame(df3.groupby('director')['avg_vote'].mean()).reset_index()
df5=df4.sort_values(by=['avg_vote']).tail(10)
df5


# In[16]:


plt.figure(figsize=(20, 12))
sns.barplot( x=df5.director , y=df5.avg_vote )
plt.title('Top 10 Directors in netflix')
plt.show() #displaying top 10 director values


# In[17]:


netflix_df_movies_only = df3[df3['type'] == 'Movie']
netflix_df_movies_only # creating a dataframe with type filtered as movies


# In[18]:


netflixMovies = netflix_df_movies_only[['type','title','country','genre','rating','release_year','duration','avg_vote']]
netflixMovies


# In[19]:


netflixMovies['duration']=netflixMovies['duration'].str.replace('min' , '')
netflixMovies['duration']


# In[20]:


netflixMovies['duration']= pd.to_numeric(netflixMovies['duration'] , errors='coerce')
netflixMovies['duration'] #coverting the object value to a numeric value


# In[21]:


netflixMovie=netflixMovies.drop_duplicates(subset='title', keep="last")
netflixMovie


# In[22]:


LongMovies = netflixMovie[netflixMovie['duration'] >120] #filtering movies having duration greater than 120


# In[23]:


LongMovies=LongMovies.head(10)
LongMovies


# In[24]:


x =LongMovies["title"]
y =LongMovies['duration']

plt.figure(figsize=(42,8)) 
plt.plot(x, y)
plt.xticks(np.arange(0, 20, step=1))  ## xticks change 
plt.show()


# In[25]:


ShortMovies= netflixMovie[netflixMovie['duration'] <120]
ShortMovies=ShortMovies.head(10)
ShortMovies


# In[26]:


ShortMovies['duration'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(20,8)) #distribution according to the duration
plt.show()


# In[27]:


plt.figure(figsize=(12,6))
netflixMovies["release_year"].value_counts()[:25].plot(kind="bar",color="green")
plt.title("Frequency of Movies which were released in different years and present in Netflix")


# In[28]:


ratingOrderM =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']


# In[29]:


movieRating = netflixMovie['rating'].value_counts()[ratingOrderM]
movieRating


# In[30]:


def rating_barplot(data, title, height, h_lim=None):
    fig, ax = plt.subplots(1,1, figsize=(15, 7))
    if h_lim :
        ax.set_ylim(0, h_lim)
    ax.bar(data.index, data,  color="#d0d0d0", width=0.6, edgecolor='black')

    color =  ['green',  'blue',  'orange',  'red']
    span_range = [[0, 2], [3,  6], [7, 8], [9, 11]]

    for idx, sub_title in enumerate(['Little Kids', 'Older Kids', 'Teens', 'Mature']):
        ax.annotate(sub_title,
                    xy=(sum(span_range[idx])/2 ,height),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
        ax.axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.1)

    ax.set_title(f'Distribution of {title} Rating', fontsize=20, fontweight='bold', position=(0.5, 1.0+0.03))
    plt.show()


# In[31]:


rating_barplot(movieRating,'Movie', 1200)


# 
# ## TV SHOWS DATA VISUALIZATION

# In[32]:


netflixTVshows = df3[df3['type'] == 'TV Show']
netflixTVshows


# In[33]:


netflixTVshows["seasons"]=netflixTVshows.duration.str.replace('s','').str[:-6]
netflixTVshows["seasons"].value_counts()


# In[34]:


netflixTVshows.drop(['duration'], axis = 1)


# In[35]:


netflixTVshows


# In[36]:


netflixTVshow=netflixTVshows.drop_duplicates(subset='title', keep="last")
netflixTVshow


# In[37]:


LongTvshows = netflixTVshow[netflixTVshow['seasons'] > '4']


# In[38]:


LongTvshows


# In[39]:


plt.figure(figsize=(12,6))
LongTvshows["seasons"].value_counts()[:25].plot(kind="bar",color="green",xlabel="seasons")
plt.title("Count of TV shows having more than 4 seasons ")


# In[40]:



netflixTVshow['firstgenre'] = netflixTVshow['genre'].apply(lambda x: x.split(",")[0])
netflixTVshow['firstcountry'] = netflixTVshow['country'].apply(lambda x: x.split(",")[0])


# In[41]:


netflixTVshow


# In[42]:


plt.figure(figsize=(12,6))
netflixTVshow[netflixTVshow["type"]=="TV Show"]["firstgenre"].value_counts()[:10].plot(kind="barh",color="red")
plt.title("Top 10 genres in TV shows",size=18)


# In[43]:


plt.figure(figsize=(12,6))
netflixTVshow[netflixTVshow["type"]=="TV Show"]["firstcountry"].value_counts()[:10].plot(kind="barh",color="blue")
plt.title("Top 10 countries in TV shows",size=18)


# In[ ]:





# ## Modelling the movie dataframe data and finding the accuracy of the prediction model

# In[44]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[45]:


netflixMovie


# In[46]:


netflixMovie.drop_duplicates(subset ="title",
                     keep = False, inplace = True)
netflixMovie.drop('type',axis=1,inplace= True)


# In[47]:


netflixMovie


# In[48]:


netflixMovie1=netflixMovie[['title','country','genre','avg_vote']]
netflixMovie1


# In[49]:


netflixMovie1['avg_vote'] = pd.qcut(netflixMovie1['avg_vote'], q=4, labels=[0,1,2,3]) #q-cutting the data into 4 quarantiles  and putting the data in 4 bins.
netflixMovie1['avg_vote']


# In[50]:


def onehot_encode(netflixMovie1, column, prefix): #representing the categorical variables as binary vectors to test and train the data
    netflixMovie1= netflixMovie1.copy()
    dummies = pd.get_dummies(netflixMovie1[column], prefix=prefix)
    netflixMovie1 = pd.concat([netflixMovie1, dummies], axis=1)
    netflixMovie1 = netflixMovie1.drop(column, axis=1)
    return netflixMovie1

netflixMovie1= onehot_encode(netflixMovie1, 'title', 'Title')
netflixMovie1= onehot_encode(netflixMovie1, 'country', 'Country')
netflixMovie1= onehot_encode(netflixMovie1, 'genre', 'Genre') 


# In[51]:


y = netflixMovie1.loc[:, 'avg_vote'] # selecting all the rows and avg_vote column
X = netflixMovie1.drop('avg_vote', axis=1) #dropping avg_vote column
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=20)# train-test method to predict the popularity of the movie based on avg_vote


# In[52]:


get_ipython().system('pip install imblearn')


# In[53]:


from sklearn.linear_model import LogisticRegression #logisticregression ML to predict the acuraccy of finding the popularity
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())


# In[54]:


clf = LogisticRegression()
model_res = clf.fit(X_train, y_train)
log_acc = clf.score(X_test, y_test)


# In[55]:


print("LogisticRegression acc",log_acc )


# In[56]:


from sklearn.neighbors import KNeighborsClassifier #using three other different ML algorithm to predict the accuracy of finding the popularity
from sklearn.tree import DecisionTreeClassifier


# In[57]:


knn_model = KNeighborsClassifier()
dec_model = DecisionTreeClassifier()


# In[58]:


knn_model.fit(X_train, y_train)
dec_model.fit(X_train, y_train)
knn_acc = knn_model.score(X_test, y_test)
dec_acc = dec_model.score(X_test, y_test)


# In[59]:


print("K-Nearest-Neighbors Acc.:", knn_acc)
print("Decision Tree Acc.:", dec_acc)


# In[60]:


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# In[61]:


mlpModel = MLPClassifier()
svmModel = SVC()


# In[62]:


mlpModel.fit(X_train, y_train)
svmModel.fit(X_train, y_train)
mlp_acc = mlpModel.score(X_test, y_test)
svm_acc = svmModel.score(X_test, y_test)


# In[63]:


print("Neural Network Acc.:", mlp_acc)
print("Support Vector Machine Acc.:", svm_acc)


# In[ ]:




