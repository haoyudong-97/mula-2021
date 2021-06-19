import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import json

import numpy as np
import math
import random
from pqdict import PQDict
import time

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (120000, rlimit[1]))

def compute_circle_dis(x, y, R):
    l2_dis = np.linalg.norm(x-y)
    theta = math.asin(l2_dis/2/R)
    circle_dis = 2 * theta * R
    return circle_dis

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def construct_graph(graph_features, K=5, dis=0.5, indices=None):
    G = {}
    ave = torch.mean(torch.norm(graph_features, dim=-1)).item()
    #dis_all = torch.mm(graph_features, graph_features.T)
    #_, indices = torch.topk(dis_all, K+2)
    #indices = indices.cpu().tolist()
    if indices is None:
        indices = []
        for idx in range(graph_features.shape[0]):
            curr_dis = torch.mm(graph_features[idx].unsqueeze(0), graph_features.T).squeeze()
            _, curr_idx = torch.topk(curr_dis, K+2)
            indices.append(curr_idx.cpu().tolist())
        return indices
    for i, index in enumerate(indices):
        # Connect each point to at most K nns
        tmp = {}
        for single_index in index[1:]:
            # don't link image to its corresponding caption
            if abs(i-single_index) == len(graph_features) // 2:
                continue
            circle_dis = compute_circle_dis(graph_features[i], graph_features[single_index], ave)
            if circle_dis > dis:
                break
            tmp[single_index] = circle_dis
            if len(tmp) == K:
                break
        G[i] = tmp
    return G

def full_dijkstra(G, start):
    inf = float('inf')
    D = {start: 0}
    Q = PQDict(D)
    P = {}

    while 1:
        if len(Q) == 0:
            break
        (v, d) = Q.popitem()
        D[v] = d
        
        for w in G[v]:
            # don't search itself
            if v != start and abs(w-P[v]) == len(G)//2:
                continue
            d = D[v] + G[v][w]
            if w not in D or D[w] > d:
                Q[w] = d
                P[w] = v
    return D, P

def get_path(paths, start, end):
    v = end
    path = [v]
    while v != start:
        v = paths[v]
        path.append(v)
    path.reverse()
    return path

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('# evaluate images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    visualizer = Visualizer(opt)

    total_steps = 0
    epoch_start_time = time.time()
    iter_data_time = time.time()
    
    # store inputs
    image_paths, captions, labels = [], [], []

    # store features
    image_features, text_features = [], []
    
    first_run = False
    
    if first_run:
    #if 0:
        curr = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batch_size

            model.set_input(data)
            with torch.no_grad():
                model.forward()
            
            for p in data['path']:
                image_paths.append(p)
                
            for c in data['T_S']:
                captions.append(c)

            for l in data['L']:
                labels.append(l)

            for ia, ta in zip(model.image_A_feature.cpu().tolist(), model.text_A_feature.cpu().tolist()):
                image_features.append(ia)
                text_features.append(ta)
            
            #for ib, tb in zip(model.image_B_feature.cpu().tolist(), model.text_B_feature.cpu().tolist()):
            #    image_features.append(ib)
            #    text_features.append(tb)

        image_features = torch.tensor(image_features)
        text_features = torch.tensor(text_features)
    
        with open('path.json', 'w+') as f:
            json.dump(image_paths, f)
        with open('captions.json', 'w+') as f:
            json.dump(captions, f)

        torch.save(labels, 'labels.pt')
        torch.save(image_features, 'if.pt')
        torch.save(text_features, 'tf.pt')

    else:
        with open('path.json', 'r') as f:
            image_paths = json.load(f)
        with open('captions.json', 'r') as f:
            captions = json.load(f)

        labels = torch.load('labels.pt')
        image_features = torch.load('if.pt')
        text_features = torch.load('tf.pt')
    
    # select images as gt
    images_per_label = {}
    for i, label in enumerate(labels):
        label = label.item()
        if label not in images_per_label:
            images_per_label[label] = [i]
        else:
            images_per_label[label].append(i)

    selected_images = []
    if 0:
        selected_thres = 3

        for k in images_per_label:
            if len(images_per_label[k]) < selected_thres:
                selected_images += images_per_label[k]
            else:
                selected_indices = random.sample(list(range(len(images_per_label[k]))), selected_thres)
                for i in selected_indices:
                    selected_images.append(images_per_label[k][i])
        # Only run once to select images
        with open('ade20k_labeled_images.txt', 'w+') as f:
            for si in selected_images:
                f.write(image_paths[si])
                f.write('\n')
    else:
        # ade20k
        with open('ade20k_labeled_images.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        for i, p in enumerate(image_paths):
            if p in lines:
                selected_images.append(i)

    # coco2017
    #selected_thres = 100
    #for k in images_per_label:
    #    if len(images_per_label[k]) < selected_thres:
    #        selected_images += images_per_label[k]
    #    else:
    #        selected_indices = random.sample(list(range(len(images_per_label[k]))), selected_thres)
    #        for i in selected_indices:
    #            selected_images.append(images_per_label[k][i])
    ### Only run once to select images
    #with open('coco2017_labeled_images.txt', 'w+') as f:
    #    for si in selected_images:
    #        f.write(image_paths[si])
    #        f.write('\n')

    # oepnimage
    #selected_thres = 5
    #for k in images_per_label:
    #    if len(images_per_label[k]) < selected_thres:
    #        selected_images += images_per_label[k]
    #    else:
    #        selected_indices = random.sample(list(range(len(images_per_label[k]))), selected_thres)
    #        for i in selected_indices:
    #            selected_images.append(images_per_label[k][i])
    ## Only run once to select images
    #with open('openimage_labeled_images.txt', 'w+') as f:
    #    for si in selected_images:
    #        f.write(image_paths[si])
    #        f.write('\n')

    print('%s images are provided with labels. There are %s label classes.' % (len(selected_images), len(images_per_label)))

    # Euclidean distance
    curr = time.time()

    #if first_run:
    point_neighbors_direct = {}
    image_features_selected = image_features[selected_images]
    for i, im in enumerate(image_features):
        if i in selected_images:
            continue
        dis = torch.norm(im - image_features_selected, dim=-1)
        tmp = {}
        dis_value, dis_indices = torch.sort(dis)
        for d, idx in zip(dis_value, dis_indices):
            tmp[selected_images[idx]] = d.item()
        point_neighbors_direct[i] = tmp
        
    with open('point_neighbor_direct.json', 'w+') as f:
        json.dump(point_neighbors_direct, f)
    #else:
    #    with open('point_neighbor_direct.json', 'r') as f:
    #        point_neighbors_direct = json.load(f)

    print('construct eul neighbor takes ' + str(time.time()-curr))

    # --------- evaluate ----------
    valid_index = []
    for it, neighbors in enumerate([point_neighbors_direct]):
        acc_1, acc_5, acc_10 = 0, 0, 0
        found = 0
        for i in neighbors:
            i = int(i)
            curr_label = labels[i]
            curr_neighbor = neighbors[i]
            if len(curr_neighbor) == 0: continue
            found += 1
            dis, index = [], []
            for k in curr_neighbor:
                dis.append(curr_neighbor[k])
                index.append(k)
            dis, index = np.array(dis), np.array(index)
            top_index = index[np.argsort(dis)]
            pred_labels = [labels[t] for t in top_index]
            if curr_label in pred_labels[:1]:
                acc_1 += 1
            if curr_label in pred_labels[:5]:
                acc_5 += 1
            if curr_label in pred_labels[:10]:
                acc_10 += 1
        print('baseline r1: %s, r5: %s, r10: %s. Found %s' % (acc_1/found, acc_5/found, acc_10/found, found))
    
    curr = time.time()
    graph_features = torch.cat([image_features, text_features], dim=0)
    graph_features /= torch.norm(graph_features[0], dim=-1)
    if first_run:
        indices = construct_graph(graph_features)
        with open('indices.json', 'w+') as f:
            json.dump(indices, f)
    else:
        with open('indices.json', 'r') as f:
            indices = json.load(f)
        
    print('construct indces takes %s' % str(time.time()-curr))

    # Geodesic distance
    #for dis in [0.1,0.15,0.2,0.25,0.3]:
    for dis in [0.31,0.32,0.33,0.34,0.35]:
    #for dis in [0.45]:
        #dis += 0.2
        K = 3
        curr = time.time()
        G = construct_graph(graph_features, K=K, dis=dis, indices=indices)
        print('construct graph takes %s with k=%s and dis=%s' % (str(time.time()-curr), K, dis))
        ave_length = 0
        for k in G:
            ave_length += len(G[k])
        print('average length is ' + str(ave_length/len(G)))
        if ave_length == 0:
            continue

        curr = time.time()
        point_neighbors_geo = {}
        for i in range(len(image_features)):
            if i != 0 and i % 1000 == 0:
                print('%s/%s takes %s' % (i, len(image_features), time.time()-curr))
                curr = time.time()
            if i in selected_images:
                continue
            dis_set, _ = full_dijkstra(G, i)
            tmp = {}
            for k in dis_set:
                if k in selected_images:
                    tmp[k] = dis_set[k]
            point_neighbors_geo[i] = tmp
        print('construct geo neighbor takes ' + str(time.time()-curr))

        # --------- evaluate ----------
        valid_index = []
        for it, neighbors in enumerate([point_neighbors_geo, point_neighbors_direct]):
            acc_1, acc_5, acc_10 = 0, 0, 0
            found = 0
            for i in neighbors:
                curr_label = labels[i]
                curr_neighbor = neighbors[i]
                if len(curr_neighbor) == 0: continue
                if it == 0:
                    valid_index.append(i)
                if it == 1 and i not in valid_index:
                    continue
                found += 1
                dis, index = [], []
                for k in curr_neighbor:
                    dis.append(curr_neighbor[k])
                    index.append(k)
                dis, index = np.array(dis), np.array(index)
                top_index = index[np.argsort(dis)]
                pred_labels = [labels[t] for t in top_index]
                if curr_label in pred_labels[:1]:
                    acc_1 += 1
                if curr_label in pred_labels[:5]:
                    acc_5 += 1
                if curr_label in pred_labels[:10]:
                    acc_10 += 1
            if found == 0:
                print('not found')
            else:
                print('r1: %s, r5: %s, r10: %s. Found %s' % (acc_1/found, acc_5/found, acc_10/found, found))
        
        if len(valid_index) > 0:
            with open('ade20k_valid_index.json', 'w+') as f:
                valid_path = [image_paths[i] for i in valid_index]
                json.dump(valid_path, f)
