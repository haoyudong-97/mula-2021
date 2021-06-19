import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import json
import nltk
import numpy as np

class QueryDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = opt.dataroot
        self.phase = opt.phase
        self.qlength = opt.qlength
        self.alength = opt.alength

        self.query_info = json.load(open(os.path.join(self.dir, 'query.%s.json' % opt.phase), 'r'))
        self.attr_info = json.load(open(os.path.join(self.dir, 'id2attr.%s.json' % opt.phase), 'r'))
        
        transform_params = get_params(self.opt, (opt.crop_size, opt.crop_size))
        self.transform = get_transform(self.opt) #, transform_params, grayscale=False)

        # Build w2i and i2w
        word_count = {}
        for q in self.query_info:
            for w in nltk.word_tokenize(q['captions']):
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1
        for k in self.attr_info:
            for w in nltk.word_tokenize(self.attr_info[k]):
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1

        self.w2i, self.i2w = {'PAD':0}, {0:'PAD'}
        for i, k in enumerate(word_count.keys()):
            self.w2i[k] = i+1
            self.i2w[i+1] = k
        print('vocab size = ' + str(len(self.w2i)))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        query = self.query_info[index]
        # FASHION 
        A_name = query['candidate']
        A = Image.open(os.path.join(self.dir, 'images/'+A_name)).convert('RGB')
        A = self.transform(A)
        A_attr = self.attr_info[A_name]
        A_attr = self.sent2num(A_attr)

        B_name = query['target']
        B = Image.open(os.path.join(self.dir, 'images/'+B_name)).convert('RGB')
        B = self.transform(B)
        B_attr = self.attr_info[B_name]
        B_attr = self.sent2num(B_attr)

        query_sent = self.sent2num(query['captions'], False)

        return {'A': A, 'B': B, 'T_A': A_attr, 'T_B': B_attr, 'Q': query_sent}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.query_info)
    
    def sent2num(self, sent, attr=True):
        length = self.alength if attr else self.qlength
        sent = [self.w2i[w] for w in nltk.word_tokenize(sent)][:length]
        sent += [0] * (length - len(sent))
        return np.array(sent)
