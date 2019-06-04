# Predicting Genre from Song Lyrics

#### Project Goal

Use several unsupervised learning approaches to attempt to predict the genre of a song from its lyrics. Labeling each artist's genre by hand is very tedious, so any automation would save time and effort.

#### Implementation and Training

I applied both TFIDF and CountVectorizer natural language processing techniques to create feature spaces out of a corpus of song lyrics. I then used K-Means, Mean Shift, and Spectral Clustering to try to detect underlying patterns to the lyrics.

In an attempt to find a technique better suited for the sparse data, I implemented Latent Dirichlet allocation and Non-negative Matrix Factorization. Using pyLDAvis, I visualized the topics.

In the last section, I used supervised models to determine to what extent the models can predict a song's genre from its lyrics.

#### Results

- None of the initial unsupervised approaches surfaced any distinct clusters.
- While the pyLDAvis did display seven distinct topics, they do not appear to fit with the seven different genres.
- Similarly, NMF was not able to match the seven topics to the seven genres.
- I was surprised to see that each of the supervised approaches performed well above the baseline in predicting genre.

#### Next Steps

- Including more lyrics into the dataset may help the models improve. I only included the top twenty artists, but there are over 50,000 more songs in the dataset. The downside here would be the need to classify each by hand to provide some ground truth.
- Reducing what I am trying to classify may improve performance. Making the classification task binary between very different types of music may be an easier task for the models to perform.
- Adjusting the feature space to include other, non-word-specific features such as song length or even number of verses or choruses in a song may improve clustering.
