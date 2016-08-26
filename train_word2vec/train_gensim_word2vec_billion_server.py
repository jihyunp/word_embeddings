from BillionWordData import BillionWordData

data_root_dir = '/extra/jihyunp0/research/word_embeddings/data'
w2v_file = '/extra/jihyunp0/research/word_embeddings/result/billion_word2vec_sg.vectors'

billion_w2v = BillionWordData(data_dir=data_root_dir, word2vec_file=w2v_file, binary=False,
                              vdim=200, window=5, sg=True, min_count=20, workers=8)

