from BillionWordData import BillionWordData

w2v_file = './billion_word2vec.vectors'
billion_w2v = BillionWordData(data_dir='./data', word2vec_file=w2v_file, binary=False,
                              vdim=200, window=5, sg=True, min_count=20, workers=3)
