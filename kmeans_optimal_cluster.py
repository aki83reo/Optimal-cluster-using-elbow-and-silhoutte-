# Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pandas as pd

def preprocessing(data_path):
    """
     IN THIS FUNCTION  WE REMOVE PUNCTUATIONS AND DONE LOWER CASE OF THE DOCUMENTS
    :param data_path: 2 columns on this data ,'transcript' and 'url'
    :return: 2 lists, preprocessed 2 columns , transcript  and url 
    """

    # read the data
    ted_talk_data = pd.read_csv(data_path)
    print(ted_talk_data)
    # get the columns
    columns = ted_talk_data.columns

    names = ted_talk_data.url
    all_names_with_title = []
    for i in range(names.__len__()):
        all_names_with_title.append(names[i][26:])


    ted_talk_data['talker_name_and_title'] = all_names_with_title


    # Remove punchuations from the column transcript which contains  our  documents of data .
    ted_talk_data['rmve_punc_data'] = ted_talk_data['transcript'].str.replace('[^\w\s]', '')

    # Change case  of the  documents  to lower case .
    ted_talk_data['rmve_punc_data'] = ted_talk_data['rmve_punc_data'].str.lower()

    # Took  out 2  imp columns  important for us
    transformed_data = ted_talk_data[['rmve_punc_data', 'talker_name_and_title']]

    # Converting  the 2 columns into  lists  for   applying seperately  embedding .
    list_sent = transformed_data['rmve_punc_data'].tolist()
    list_title = transformed_data['talker_name_and_title'].tolist()

    return list_sent,list_title


def tfidf_conversion(list_data):
    # Initialize TFIDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, ngram_range=(1, 3))



    #Apply  our tf-idf  vectorizer  in our  documents

    tfidf_matrix = tfidf_vectorizer.fit_transform(list_data)

    # terms is just a list of the features used in the tf-idf matrix. This is a vocabulary.

    terms = tfidf_vectorizer.get_feature_names()

    return tfidf_matrix,terms


def kmeans(num_clusters,tfidf_matrix,modal):

    km = KMeans(n_clusters=num_clusters)
    # Fitting our document into  kmeans
    km.fit(tfidf_matrix)
    # Putting  all  cluster into  list
    clusters = km.labels_.tolist()

    # store our  model
    joblib.dump(km, modal)
    # Defining centroids  of every clusters
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    return clusters,order_centroids

##### Elbow Method
def optimal_cluster_elbow_method(matrix_data,max_clusters_check):
    ############# Get optimal clusters
    '''
    k , here we are taking max value . 
    -->> For each k value we will initialize k-means and use the inertia attribute to identify the sum of
     square distances of samples to the nearest cluster centre.
    
    -->> As k increases, the sum of squared distance tends to zero. Imagine we set k to its maximum value n (where n is number of samples) 
     each sample will form its own cluster meaning sum of squared distances equals zero.
      
    '''

    Sum_of_squared_distances=[]
    K = range(1,max_clusters_check)            ####  15 was  taken  arbitaorily  to see  where  our elbow touches
    for k  in K:
        km=KMeans(n_clusters=k)
        km=km.fit(matrix_data)
        Sum_of_squared_distances.append(km.inertia_)

    '''
    Below is a plot of sum of squared distances for k in the range specified above. If the plot looks like an arm, then the elbow on the arm is optimal k.
    
    '''

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

##### Silhoutte method
def optimal_cluster_silhouette_method(matrix_data,max_clusters_check):
    #### Silhoutte  method

    '''
    The silhouette plot displays a measure of how close each point in one cluster is
    to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually.

    Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters.
    A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.

    '''
    for n_cluster in range(2,max_clusters_check):
        kmeans = KMeans(n_clusters=max_clusters_check).fit(matrix_data)
        label = kmeans.labels_
        sil_coeff = silhouette_score(matrix_data, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


def model_validation(model,list_title,num_clusters,terms, clusters, order_centroids):
    """
    THIS  FUNCTION WE  ARE LOADING  OUR MODEL  AND  PRINTING 5 CLUSTERS OF SIMILAR SPEAKERS
    :param model: 
    :param list_title: It  contains the list of speaker names with  topics 
    :param num_clusters:
    :param terms: 
    :param clusters:
    :param order_centroids:
    :return:
    """

    # Extract  our model to use
    km = joblib.load(model)
    # Putting our  title data into a dataframe
    last_df = {'title': list_title}
    results = pd.DataFrame(last_df, index=[clusters])

    # This below code will give top 5 clusters of
    # auretors  having similar  speech

    for i in range(num_clusters):
        print("Cluster %d titles:" % i, end='')
        for title in results.ix[i]['title'].values.tolist()[1:num_clusters]:
            print(' %s,' % title, end='')

    # This  will  give  5  clusters ,
    # in each clusters top 5 words which are similar.
    for i in range(num_clusters):
        print("Cluster %d:" % i)
        for ind in order_centroids[i,:num_clusters]:  ##### You can change it  and see  how many values you want to see in each  cluster
            print(' %s' % terms[ind])


# Preprocessing
data_path = "E://personal//datasets//transcripts.csv"
list_sent, list_title = preprocessing(data_path)


## Conversion tf-idf
conversion=tfidf_conversion(list_sent)

##  Finding optimal clusters of dataset  using  elbow method
optimal_cluster_elbow_method(conversion[0],15)

##  Finding optimal clusters of dataset  using  silhoutte method

optimal_cluster_silhouette_method(conversion[0],11)

'''
For n_clusters=2, The Silhouette Coefficient is 0.025221700503161602
For n_clusters=3, The Silhouette Coefficient is 0.025360625895740875
For n_clusters=4, The Silhouette Coefficient is 0.024858191350452728
For n_clusters=5, The Silhouette Coefficient is 0.025151911809533658
For n_clusters=6, The Silhouette Coefficient is 0.024629056036962343
For n_clusters=7, The Silhouette Coefficient is 0.02083552089015085
For n_clusters=8, The Silhouette Coefficient is 0.02414572702868102
For n_clusters=9, The Silhouette Coefficient is 0.022338255264880966
For n_clusters=10, The Silhouette Coefficient is 0.02387461430481777
'''

## From the  above graph of elbow method and silhouette method   we  took 5 clusters .


### Finding Clusters from  the dataset

# Preprocessing
data_path = "E://personal//datasets//transcripts.csv"
list_sent, list_title = preprocessing(data_path)


# Model Building
num_clusters = 5
model = 'doc_cluster.pkl'
clusters, order_centroids = kmeans(5,tfidf_conversion(list_sent)[0],model)
print("Model Building completed")


# Model Validation

terms=tfidf_conversion(list_sent)[1]
model_validation(model, list_title, num_clusters, terms, clusters, order_centroids)
print("Model Validation completed")

'''
,Cluster 0:
 actually
 things
 right
 youre
 theres
 
Cluster 1:
 new
 water
 data
 space
 actually
 
Cluster 2:

 world
 percent
 countries
 need
 country
Cluster 3:

 said
 life
 got
 did
 love
 
Cluster 4:
 women
 men
 woman
 said
 world
 
'''