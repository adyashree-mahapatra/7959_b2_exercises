# import pandas, numpy
import pandas as pd
import numpy as np
from datetime import datetime,date
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

# Create the required data frames by reading in the files
df = pd.read_excel('SaleData.xlsx')
df1 = pd.read_csv('imdb.csv',escapechar='\\')
df2=pd.read_csv('diamonds.csv')
df3=pd.read_csv('movie_metadata.csv')
df1.to_csv('imdb1.csv') 

# Q1 Find least sales amount for each item
def least_sales(df):
    ls = df.groupby(["Item"])["Sale_amt"].min().reset_index()
    return ls

# Q2 compute total sales at each year X region
def sales_year_region(df):
    df['Year'] = df['OrderDate'].apply(lambda time: time.year)
    df1 = df.groupby(by=['Year','Region'])['Sale_amt'].sum().reset_index()
    return df1
    
# Q3 append column with no of days difference from present date to each order date
def days_diff(df):
    today = datetime.now().date()
    for col in df.columns:
        if df['OrderDate'].dtypes == 'M8[ns]':
            df['OrderDate'] = df['OrderDate'].dt.date
    df['days_diff'] = (today- df['OrderDate'])
    return df
    
# Q4 get dataframe with manager as first column and  salesman under them as lists in rows in second column.
def mgr_slsmn(df):
    val = {}
    for i in df["Manager"].unique():
      val[i]=df.loc[df["Manager"]==i,"SalesMan"]
      val[i]=list(set(val[i])) 
    df3 = pd.DataFrame(list(val.items()),columns=["Managers","List_of_Salesmen"])       
    return df3

# Q5 For all regions find number of salesman and number of units
def slsmn_units(df):
    val={}
    for i in df["Region"].unique():
        val[i]=df.loc[df["Region"]==i,"SalesMan"]
        val[i]=list(set(val[i]))
        val[i]= len(val[i])
    ls = pd.DataFrame(list(val.items()),columns=["Region","salesmen_count"])
    d = df.groupby("Region")["Sale_amt"].sum()
    ls = pd.merge(ls,d,on="Region")
    ls = ls.rename(columns={'Sale_amt':"total_sales"})
    return ls


# Q6 Find total sales as percentage for each manager
def sales_pct(df):
    t = df["Sale_amt"].sum()
    d = df.groupby("Manager")["Sale_amt"].sum().reset_index()
    d['Sale_amt']= (d['Sale_amt']/t)*100
    d=d.rename(columns={"Sale_amt":"percent_sales"})
    return d


# Q7 get imdb rating for fifth movie of dataframe
def fifth_movie(df):
    ls = df.at[4,'imdbRating']
    return ls

# Q8 return titles of movies with shortest and longest run time
def movies(df):
    ls = df[df.duration==df.duration.min()]['title']
    ms = df[df.duration==df.duration.max()]['title']
    return ls,ms

# Q9 sort by two columns - release_date (earliest) and Imdb rating(highest to lowest)
def sort_df(df):
    df1 = df.sort_values(['year','imdbRating'],ascending=[False,False])
    return df1


# Q10 subset revenue more than 2 million and spent less than 1 million & duration between 30 mintues to 180 minutes
def subset_df(df):
    df1 = df[(df['gross'] > 20000000) & (df['budget'] < 10000000) & (df['duration'] >= 30) & (df['duration'] <= 180)]
    return df1


# Q11 count the duplicate rows of diamonds DataFrame.
def dupl_rows(df):
    df1 = df.drop_duplicates(keep='first')
    return df.shape[0]-df1.shape[0]

# Q12 droping those rows where any value in a row is missing in carat and cut columns
def drop_row(df):
    df2 = df.dropna(axis=0,subset=['carat','cut'])
    return df2

# Q13 subset only numeric columns
def sub_numeric(df):
    df3 = df._get_numeric_data()
    return df3

# Q14 compute volume as (x*y*z) when depth > 60 else 8
def volume(df):
    df['z'] = pd.to_numeric(df.z, errors='coerce')
    df4 = pd.DataFrame(columns=["Volume"])
    df4['Volume']= np.where(df['depth']>=60,df.x*df.y*df.z,8)
    return df4

# Q15 impute missing price values with mean
def impute(df):
    m=df['price'].mean()
    df['price'].fillna(value=m,inplace=True)
    return df

#Bonus_Problem

#1. Generate a report that tracks the various Genere combinations for each type year on year. The result data frame should contain type, Genere_combo, year, avg_rating, min_rating, max_rating,total_run_time_mins

def gt_k(my_d):
    for key,value in my_d.items():
        s=[]
        for i,j in value.items():
            if(j>0):
                s.append(i[0])
        genre_dic[key]=s
    return genre_dic


def bonus_01(df):
    grp=df.groupby(['type','year']).agg([np.sum]) 
    grp1=grp.loc[:,"Action":].transpose()
    grp_dict=grp.to_dict()
    genre_dic={}     
    genre_dic=gt_k(grp1)
    s = pd.Series(genre_dic)
    
    movie_data = df.groupby(['type','year']).agg({'imdbRating': [min,max, np.mean],'duration':(sum)})
    movie_data['Genre_combo']= s
    movie_data['duration']=movie_data['duration']/60
    movie_data=movie_data.rename(columns={"min": "min_rating", "max": "max_rating","mean":"avg_rating","duration":
                                          "total_run_time_mins","sum":"","imdbRating":"Rating"})
    return movie_data

#2. Is there a realation between the length of a movie title and the ratings ? Generate a report that captures the trend of the number of letters in movies titles over years. We expect a cross tab between the year of the video release and the quantile that length fall under. The results should contain year, min_length, max_length, num_videos_less_than25Percentile, num_videos_25_50Percentile ,num_videos_50_75Percentile, num_videos_greaterthan75Precentile

def per_25(df,cr):
    res = sum(1 for i in df if i>0 and i<=cr) 
    return res

def per(df,cr1,cr2):
    res = sum(1 for i in df if i>cr1 and i<=cr2) 
    return res

def bonus_02(df):
    df['Length']=df['wordsInTitle'].str.len()
    df2=pd.DataFrame(columns=['Len','Rate'])
    
    df2['Len']=df['Length']
    df2['Rate']=df['imdbRating']
    df2['Rate'].fillna(0,inplace=True)
    df2['Len'].fillna(0,inplace=True)
    df2['Len/10']=df2['Len']/10
    print(df[['Length','imdbRating']].corr())
    print("\n")
    df['quantile']=df['Length'].quantile()
    print(pd.crosstab(df['year'],df['quantile']))
    print("\n")
    d=pd.DataFrame()
    d=df[['year', 'Length']].groupby('year').quantile().reset_index().rename(columns={"Length": "quantile"})
    print(pd.crosstab(d['year'],d['quantile']))
    print("\n")
    kmeans = KMeans(n_clusters=4).fit(df2)
    y_kmeans= kmeans.predict(df2)
    plt.scatter(df2.iloc[:, 2], df2.iloc[:, 1], c=y_kmeans, s=10,cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 2], centers[:, 1], c='black', s=20, alpha=1.0)
    plt.xlabel('Length')
    plt.ylabel('imdbRating')
    plt.title('Length of movie title Vs Its imdbRating')

    b=pd.DataFrame(columns=
                   ['no_of_movies','maximum_length','minimum_length','25_percentile','50_percentile','75_percentile','100_percentile'])
    df['Length'].fillna(0,inplace=True)
    b['no_of_movies']=df.groupby('year')['Length'].count()

    b['maximum_length']=df.groupby('year')['Length'].max()
    b['minimum_length']=df.groupby('year')['Length'].min()
    b['25_percentile']=df.groupby('year')['Length'].apply(lambda x:per_25(x,np.percentile(x,25)))
    b['50_percentile']=df.groupby('year')['Length'].apply(lambda x:per(x,np.percentile(x,25),np.percentile(x,50)))
    b['75_percentile']=df.groupby('year')['Length'].apply(lambda x:per(x,np.percentile(x,50),np.percentile(x,75)))
    b['100_percentile']=df.groupby('year')['Length'].apply(lambda x:per(x,np.percentile(x,75),np.percentile(x,100)))
    
    return b

#3. In diamonds data set Using the volumne calculated above, create bins that have equal population within them. Generate a report that contains cross tab between bins and cut. Represent the number under each cell as a percentage of total.
def volume(df):
    df['z'] = pd.to_numeric(df.z, errors='coerce')
    return np.where(df['depth']>=60,df.x*df.y*df.z,8)

def bonus_03(df):
    a=volume(df)
    df['bin'] = pd.qcut(a, q=6)
    df1=np.array(df['bin'].value_counts())
    t=pd.crosstab(df.bin,df.cut)
    print(t,"\n")
    i=0
    r=pd.DataFrame()
    for i in range(len(df1)):
        r[i]=(t.loc[i]/df1[i])*100
    return r.transpose()

#4. Generate a report that tracks the Avg. imdb rating quarter on quarter, in the last 10 years, for movies that are top performing. You can take the top 10% grossing movies every quarter. Add the number of top performing movies under each genere in the report as well.

def bonus_04(df):
    df=df.sort_values(by=['title_year','gross'],ascending=False)
    df1=df.groupby("title_year").agg({'title_year':'count'})
    df1.columns=['count']
    df1=df1.sort_values(by=['title_year'],ascending=False)
    n=10
    x=0
    y=0
    ans=pd.DataFrame()
    for x in range(0,10):
        res=pd.DataFrame()
        df2=df.iloc[y:y+int(df1.iloc[x])]
        res['year']=df2.head(int(df1.iloc[x]*(n/100)))['title_year']
        res['movies_title']=df2.head(int(df1.iloc[x]*(n/100)))['movie_title']
        res['genres']=df2.head(int(df1.iloc[x]*(n/100)))['genres']
        res['gross']=df2.head(int(df1.iloc[x]*(n/100)))['gross']
        y=y+int(df1.iloc[x])
        ans=pd.concat([ans,res])  
    return ans
        

#5. Bucket the movies into deciles using the duration. Generate the report that tracks various features like nomiations, wins, count, top 3 geners in each decile.
def bonus_05(df):
    df['deciles']=pd.qcut(df["duration"], 10, labels=False)
    gen=df.groupby('deciles').agg([np.sum])
    gen=gen.loc[:,"Action":]
    x=0
    b=[]
    for y in range(0,10):
        gen1=gen.sort_values(by=x,axis=1, ascending=False)
        gen1.columns = gen1.columns.map('_'.join)
        a=list(gen1.columns[:3])
        b.append(a)
        x=x+1

    df1=pd.Series(b)

    df2=df.groupby('deciles').agg({'nrOfNominations':'sum','nrOfWins':'sum','imdbRating':'count'}).rename(columns= {'nrOfNominations':'nomination','nrOfWins':'wins','imdbRating':'count'}) 
    df2['top3_genres']=df1
    return df2
    