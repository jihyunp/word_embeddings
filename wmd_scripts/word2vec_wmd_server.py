from RedditData import RedditData
from Word2VecUtils import plot_score_histogram, plot_three_scores_hist

# CBOW model
data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/gensim_word2vec_2008_2015.vectors'
reddit_data = RedditData(data_dir, word2vec_file)

n_test = 1000
within_subreddit_cbow, within_post_cbow, random_doc_cbow, comments_cbow = reddit_data.get_three_scores(n_test)

plot_score_histogram(within_subreddit_cbow, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post_cbow, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc_cbow, 'with Random Docs', 'wmd_hist_random_doc.pdf')

plot_three_scores_hist([within_subreddit_cbow, within_post_cbow, random_doc_cbow],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       'wmd_three_hist_cbow.pdf')

# Skip gram
word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/reddit_word2vec_sg/gensim_word2vec_sg_2008_2015.vectors'
reddit_data_sg = RedditData(data_dir, word2vec_file)

n_test = 1000
within_subreddit, within_post, random_doc, comments_sg = reddit_data_sg.get_three_scores(n_test)

plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')

plot_three_scores_hist([within_subreddit, within_post, random_doc],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       'wmd_three_hist_sg.pdf')
