import torch
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--glove_file', type=str, default='glove.6B.300d.txt')
parser.add_argument('--data_dir', type=str, default='dataset/')
parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--rm_stopwords', type=bool, default=False, help='Remove stopwords in global word Node')
parser.add_argument('--lower', dest='lower', action='store_true', default=True, help='Lowercase all words.')
parser.add_argument('--min_freq', type=int, default=0, help='word min frequent')
parser.add_argument('--batch_size', type=int, default=16, help='batch size cuda can support')

args = parser.parse_args()



class Config:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.ws_edge_bucket = 10
        self.wn_edge_bucket = 10
        self.wp_edge_bucket = 10
        self.train_f, self.val_f, self.test_f = (os.path.join(self.data_dir, o) for o in ['train.json', 'dev.json', 'test.json'])
        self.glove_f = os.path.join(self.data_dir, self.glove_file)
        self.embed_f = os.path.join(self.data_dir, 'embeddings.npy')
        self.proce_f = os.path.join(self.data_dir, 'dataset_preproc.p')
        self.proce_f_c = os.path.join(self.data_dir, 'dataset_preproc_c.p')
        self.num_workers = self.batch_size//2
        self.gpus = 1
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = Config(args)
