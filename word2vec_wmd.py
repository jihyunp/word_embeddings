from random import sample
from RedditData import RedditData
import matplotlib.pyplot as plt

data_dir = './reddit_data_MH/2015'
reddit_data = RedditData(data_dir)
n_test = 500

# Select random test set
n_data = reddit_data.n_comments
test_idx = sample(range(n_data), n_test)

within_subreddit = []
within_post = []
random_doc = []


for idx in test_idx:
    single_data = reddit_data.json_data[idx]
    subreddit, link_id = reddit_data.extract_subreddit_link_info(single_data)
    sent_to_compare = reddit_data.get_parsed_doc(single_data['body'])
    if len(sent_to_compare) < 2:
        continue
    doc1 = reddit_data.get_random_doc_within_subreddit(subreddit)
    doc2 = reddit_data.get_random_doc_within_post(link_id)
    doc3 = reddit_data.get_random_doc_outside_subreddit(subreddit)
    if len(doc1) < 2:
        continue
    if len(doc2) < 2:
        continue
    if len(doc3) < 2:
        continue
    within_subreddit.append(reddit_data.get_wmdistance(sent_to_compare, doc1))
    within_post.append(reddit_data.get_wmdistance(sent_to_compare, doc2))
    rdist = reddit_data.get_wmdistance(sent_to_compare, doc3)
    random_doc.append(rdist)




