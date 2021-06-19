import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import json
import nltk
import numpy as np

class ImagetextDataset(BaseDataset):
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
        if 'coco' in self.dir or 'openimage' in self.dir:
            self.sent = True
        else:
            self.sent = False
        self.phase = opt.phase
        self.qlength = opt.qlength
        self.alength = opt.alength
        
        self.image_root = os.path.join(opt.dataroot, 'images/training_full')

        if opt.supervised:
            target_names = []
            with open('ade20k_labeled_images.txt', 'r') as f:
            #with open('openimage_labeled_images.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('/')[-1]
                    target_names.append(line)
                if opt.val:
                    self.image_names = [n for n in os.listdir(self.image_root) if '.jpg' in n]
                    self.image_names = [n for n in self.image_names if n not in target_names]
                else:
                    self.image_names = target_names
        else:
            self.image_names = [n for n in os.listdir(self.image_root) if '.jpg' in n]

        self.attr_info = json.load(open(os.path.join(self.dir, 'id2attr.json'), 'r'))
        self.label_info = json.load(open(os.path.join(self.dir, 'id2label.json'), 'r'))

        # check if image names appear in label & caption
        self.image_names = [name for name in self.image_names if name in self.label_info.keys()]
        self.image_names = [name for name in self.image_names if name in self.attr_info.keys()]

        transform_params = get_params(self.opt, (opt.crop_size, opt.crop_size))
        self.transform = get_transform(self.opt) #, transform_params, grayscale=False)

        # Build w2i and i2w
        word_count = {}
        for k in self.attr_info:
            if not self.sent:
                for w in self.attr_info[k]:
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1
            else:
                for w in nltk.word_tokenize(self.attr_info[k]):
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1

        self.w2i, self.i2w = {'PAD':0}, {0:'PAD'}
        for i, k in enumerate(word_count.keys()):
            self.w2i[k] = i+1
            self.i2w[i+1] = k

        # Build label to int
        self.label2id = {}
        count = 0
        for k in self.label_info:
            label = self.label_info[k]
            if label not in self.label2id:
                self.label2id[label] = count
                count += 1

        print('vocab size = ' + str(len(self.w2i)))
        print('# of class ' + str(len(self.label2id)))

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
        curr_path = self.image_names[index]
        path = os.path.join(self.image_root, curr_path)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
            
        sent = self.attr_info[curr_path]
        if not self.sent:
            sent_str = ""
            for s in sent:
                sent_str += s + " "
            sent_str = sent_str[:-1]
            sent_index = [self.w2i[w] for w in sent][:self.alength]
        else:
            sent_str = sent
            sent_index = nltk.word_tokenize(sent_str)[:self.alength]
            sent_index = [self.w2i[w] for w in sent_index]
        
        sent_index += [0] * (self.alength - len(sent_index))
        sent_index = np.array(sent_index)
        label = self.label_info[curr_path]
        label = self.label2id[label]

        return {'A': img, 'path': path, 'T_A': sent_index, 'id': curr_path, 'L': label, 'T_S': sent_str}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_names)
    
    def sent2num(self, sent, attr=True):
        length = self.alength if attr else self.qlength
        sent = [self.w2i[w] for w in sent.split()][:length]
        sent += [0] * (length - len(sent))
        return np.array(sent)
