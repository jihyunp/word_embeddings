from RedditData import RedditData
from Word2VecUtils import plot_score_histogram, plot_three_scores_hist
from datetime import datetime

# CBOW model
# data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
# word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/gensim_word2vec_2008_2015.vectors'
# reddit_data = RedditData(data_dir, word2vec_file)
#
# n_test = 1000
# within_subreddit_cbow, within_post_cbow, random_doc_cbow, comments_cbow = reddit_data.get_three_scores(n_test)
#
# plot_score_histogram(within_subreddit_cbow, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
# plot_score_histogram(within_post_cbow, 'within Post', 'wmd_hist_within_post.pdf')
# plot_score_histogram(random_doc_cbow, 'with Random Docs', 'wmd_hist_random_doc.pdf')
#
# plot_three_scores_hist([within_subreddit_cbow, within_post_cbow, random_doc_cbow],
#                        ['within Subreddit', 'within Post', 'with Random Docs'],
#                        'wmd_three_hist_cbow.pdf')


""" Skip gram """

data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
# word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/reddit_word2vec_sg/gensim_word2vec_sg_2008_2015.vectors'
word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/reddit_word2vec_sg/skipgram_binary.vectors'
reddit_data= RedditData(data_dir, word2vec_file, binary=True)

n_test = 2000
within_subreddit, within_post, random_doc, comments_sg = reddit_data.get_three_scores(n_test)

print('plotting histograms')
plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')

plot_three_scores_hist([within_subreddit, within_post, random_doc],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       'wmd_three_hist_sg.pdf')

print('get most and least similar comments within the post')
res1 = reddit_data.get_most_and_least_similar_comments_within_post(n_test=50, n_most=3, n_least=3,
                                                                   print_result=True)
print('done')
print(datetime.now())
# test_sent = 'distributed representations of words and phrases and their compositionality'
# res2 = reddit_data.get_most_and_least_similar_comments_custom(test_sent, n_pool=50,
#                                                              n_most=3, n_least=3,
#                                                              subreddit=None, verbose=True,
#                                                              print_result=True)
#
# comment_idx = 1179141 #1179141, 382519, 2040322
# res3 = reddit_data.get_most_and_least_similar_comments_within_post_by_idx(comment_idx)
#
#
# string1 = "they're already disadvantage missing classes. he's making anything difficult can't attend making easier can. helping students pass teacher's job."
# string2 = string1 + "comment removed includes sort insulting discriminating word. bot r automoderator comments q11pu automoderator action performed automatically. please contact moderators subreddit message compose 2fr 2fgetmotivated questions concerns."
# reddit_data.get_wmdistance_str(string1, string2)


""" billion word (news) embeddings, skipgram """
print('\nWith billion word (news) embeddings.. ')
word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/billion_word2vec_sg/billion_word2vec_sg.vectors'
reddit_data_billion_emb = RedditData(data_dir, word2vec_file, binary=False)

n_test = 2000
within_subreddit_b, within_post_b, random_doc_b, comments_sg_b = reddit_data_billion_emb.get_three_scores(n_test)

print('plotting histograms')
plot_score_histogram(within_subreddit_b, 'within Subreddit, News Embeddings', './fig/wmd_hist_within_subreddit_billion.pdf')
plot_score_histogram(within_post_b, 'within Post, News Embeddings', './fig/wmd_hist_within_post_billion.pdf')
plot_score_histogram(random_doc_b, 'with Random Docs, News Embeddings', './fig/wmd_hist_random_doc_billion.pdf')

plot_three_scores_hist([within_subreddit_b, within_post_b, random_doc_b],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       './fig/wmd_three_hist_sg_billion.pdf')

print('get most and least similar comments within the post')
res_b = reddit_data.get_most_and_least_similar_comments_within_post(n_test=50, n_most=3, n_least=3,
                                                                   print_result=True)
print('done')
print(datetime.now())





