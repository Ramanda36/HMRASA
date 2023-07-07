# HMRASA
A Hybrid Movie Recommendation Method with Aspect-Based Sentiment Analysis of Reviews

## Preprocessing
* Data preprocessing
* Get the importance(weights) of each Aspect

`test.ipynb:`  Data preprocessing. (Dataset from IMDb Movie Reviews Dataset/ MovieLens 100K)

## Movie_recommendation
* Extract Aspect-Opinion pairs from movie reviews
* Aspect-Opinion pairs clustering
* Map@k with Hybrid movie recommendation method based on sentiment analysis of tweets movie reviews

`Aspect_Opinion_pairs_extraction.ipynb:`  Extract Aspect-Opinion pairs described by each movie aspect.

`K_Medoids.ipynb:`  Clustering of Aspect Mentions. 

  * Clusters including: acting, direction, music, story and effect.

`imdb_related_movies.ipynb:`  Get related movies recommended by imdb in the movie database.

`compare_model_collaborate.ipynb:`  Implement Hybrid Model composed of Content Based Metadata and Collaborative Filtering.

`compare_model.ipynb:`  Implement a hybrid movie recommendation method based on sentiment analysis of tweets movie reviews.

  * Compared Hybrid method is combined the result of `compare_model_collaborate.ipynb` part and sentiment analysis through Vader.

## Multi-label classification
* Map@k with HMRASA/ Pure Muli-label sentiment  Method/ Collaborative Filtering Method
* Multi-label Sentiment classification
* Get the sentiment score for each movie

`weighted_sum.ipynb:`  Combined each parts of model and get the Map@k through several recommendate methods.

`opinions_sentiment.py:`  Get movies’ sentiment score.

  * Get sentiment score of each aspect. (Base on entiwordnet)
  * Get multi-label sentiment score, belongs to “story” aspecte. (Base on BERT)

`test.py:`  Define my multi-label classification model through `bert-base-uncased`.


