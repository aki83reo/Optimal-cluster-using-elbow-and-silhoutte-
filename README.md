
###   POC TO FIND WORDS FROM MULTIPLE DOCUMENTS AND CLUSTER THEM TOGETHER BY ANALYZING  OPTIMAL CLUSTERS FOR THE DATA  : 



# Data Set :  i have  used tad talk dataset from keggle.
It has 2 columns one is the documents  containing  all talks  in text format  , another will  be which author and title .

## Preporcessing  : 
I preporcessed the documents  by removing stopwords  and punctuations for model building .

## tf-idf :  

I used tf-idf  to convert our document columns  into a vector representation such  the computer can understand.



# Optimal-cluster-using-elbow-and-silhoutte-

Applied optimal Cluster algorithm by itterating  15 ,20 times and plot  to find the elbow , another  is silhoutte  method  to  score  clusters according  to  closeness of datapoints with respect to  other clusters . 


BELOW ARE THE  ELBOW  METHOD USED FOR 15 AND 20 KMEANS ITTERATIONS  :

1st  :  Elbow method (15 itterations to  plot )  

![cluster_15_elbow](https://user-images.githubusercontent.com/7880341/46594186-ea7eb600-caee-11e8-9d32-1bb47409c002.png)

With  15 itterations  was not  sure  about  the  optimal  clusters  so  tried with  20 itterations  ,  it might take some time to execute .

2nd : Elbow method (20 itterations to plot )


![cluster_20_elbow](https://user-images.githubusercontent.com/7880341/46594278-a17b3180-caef-11e8-8014-5d20e7d901b3.png)


3RD : Silhoutte  method  to  find optimal  clusters  . 

I used 11  itterations  to  find  which one will  bring the optimum  clusters  ,  below  is  the results :


For n_clusters=2, The Silhouette Coefficient is 0.025221700503161602
For n_clusters=3, The Silhouette Coefficient is 0.025360625895740875
For n_clusters=4, The Silhouette Coefficient is 0.024858191350452728
For n_clusters=5, The Silhouette Coefficient is 0.025151911809533658
For n_clusters=6, The Silhouette Coefficient is 0.024629056036962343
For n_clusters=7, The Silhouette Coefficient is 0.02083552089015085
For n_clusters=8, The Silhouette Coefficient is 0.02414572702868102
For n_clusters=9, The Silhouette Coefficient is 0.022338255264880966
For n_clusters=10, The Silhouette Coefficient is 0.023874614304817

Both the results  are not complete  satisfactory  ,  but i choosed 5 clusters as  an  mid  level  cluster  .



###  Applied kmeans clusterring  algo  (5 clusters ) , to find common words  in  every docs :

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
