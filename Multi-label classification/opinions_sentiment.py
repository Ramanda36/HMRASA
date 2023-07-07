import nltk
import csv
import glob
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from test import Bert_pre
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('omw-1.4')

def read_csv(path):
    oa_pairs = pd.read_csv(path, encoding= 'unicode_escape')
    print(oa_pairs.shape)
    print(oa_pairs)
    return oa_pairs

def cal_w_synsets_s_score(word): # opinion modifier may contains multiple words
    word = word.split()
    aspect_score_list = []
    for w in word:
        denominator = 0
        synsets_score = []
        w_list = list(swn.senti_synsets(w))
        if not w_list:
            if  w == word[-1]:
                return 0
            continue
        for i in range(len(w_list)):
            syn_pos = w_list[i].pos_score()
            syn_neg = w_list[i].neg_score()
            try:
                syn_score = (syn_pos - syn_neg) / (i + 1)
                synsets_score.append(syn_score)
                # print(synsets_score)
            except:
                pass
            denominator +=  1 / (i + 1)
        # print('denominator: ',denominator)
        w_score = sum(synsets_score) / denominator 
        # print('w_score: ',w_score)
        aspect_score_list.insert(0, w_score)
    # print(aspect_score_list)
    while len(aspect_score_list)>1:
        if aspect_score_list[0] > 0 and aspect_score_list[1] < 0:
            score = aspect_score_list[0] - (aspect_score_list[0] * abs(aspect_score_list[1]))
        else:
            score = (-1) * (abs(aspect_score_list[0]) + (1 - abs(aspect_score_list[0])) * abs(aspect_score_list[1]))
        aspect_score_list[0] = score
        aspect_score_list.pop(1)
    return aspect_score_list[0] # return sentiment score of a aspect(base on sentiwordnet)

def check_cluster_belongs(oa_pairs):
    init_center_ws = [['acting','chemistry','performanece','charm','comedian'], ['direction','director','filmmaker','screenplay','sequence','script','editing','lines'],
    ['music','audio','vocals'], ['story','plot'], ['effect','scene','efficacy','scenery','photography','camera','budget']]
    c1 = oa_pairs[oa_pairs['cluster'] == 0]
    c2 = oa_pairs[oa_pairs['cluster'] == 1]
    c3 = oa_pairs[oa_pairs['cluster'] == 2]
    c4 = oa_pairs[oa_pairs['cluster'] == 3]
    c5 = oa_pairs[oa_pairs['cluster'] == 4]
    clusters_list = [list(c1['aspect']), list(c2['aspect']), list(c3['aspect']),
    list(c4['aspect']), list(c5['aspect'])]
    cluster_belong_to_aspect = {'acting': [], 'direction': [], 'music': [], 'story': [], 'effect': []}
    for c in clusters_list:
        for init in init_center_ws:
            for i in init:
                if i in c:
                    cluster_belong_to_aspect[init_center_ws[init_center_ws.index(init)][0]].append(clusters_list.index(c))
                    break
    return cluster_belong_to_aspect
        
class Opinions_sentiment(object):
    print('opinion')
    def run_opinions_sentiment(self):
        path = '.\movies_aspect_clusterd_result_0423\*.csv'
        file_names = glob.glob(path)
        # file = '.\movies_oa_pairs\Kingpin 1996.csv'
        for file in file_names:
            file_name = file.split('\\')[-1]
            title = file_name.rsplit(' ', 1)[0]
            print('run_opinions_sentiment')
            clustered_oa_pairs = read_csv(file)
            cluster_belong_to_aspect = check_cluster_belongs(clustered_oa_pairs)
            all_clus_score = []
            for i in range(5):  # for each cluster
                aspects_score = []
                opinions = clustered_oa_pairs.loc[clustered_oa_pairs.cluster == i, 'DynamicColumn_1':clustered_oa_pairs.columns[-1]].values.flatten().tolist()
                opinions = [x for x in opinions if pd.isnull(x) == False] 
                print(opinions)
                for w in opinions:
                    aspects_score.append(cal_w_synsets_s_score(w))
                # print(len(aspects_score))
                all_clus_score.append(np.array(aspects_score).sum() / len(aspects_score))
            # print(all_clus_score)
            each_aspect_score = [0,0,0,0,0]
            c_value = list(cluster_belong_to_aspect.values())
            for cluster in c_value:
                for c in cluster:
                    each_aspect_score[c_value.index(cluster)] =  each_aspect_score[c_value.index(cluster)] + all_clus_score[c]
                if each_aspect_score[c_value.index(cluster)] != 0:
                    each_aspect_score[c_value.index(cluster)] = each_aspect_score[c_value.index(cluster)] / len(cluster)
                
            movie_csv = self.path.split('/')[-1]
            title = movie_csv.rsplit(' ', 1)[0]
            with open('all_movies_aspect_sentiment_score_0423.csv','a', newline = '') as f:
                each_aspect_score.insert(0, title) 
                writer = csv.writer(f)
                writer.writerow(each_aspect_score)
            story_clusters_Index = cluster_belong_to_aspect['story']
            story_opinions = clustered_oa_pairs.loc[clustered_oa_pairs['cluster'].isin(story_clusters_Index), 'DynamicColumn_1':clustered_oa_pairs.columns[-1]].values.flatten().tolist()
            story_opinions = opinions = [x for x in opinions if pd.isnull(x) == False] 
            bert_pre = Bert_pre(story_opinions)

            # get sentiment score of each movie through BERT multi-label calssification model. 
            with open('bert_predict_sentiment_score_0502.csv','a', newline = '') as f:
                bert_pre.insert(0, title) 
                writer = csv.writer(f)
                writer.writerow(bert_pre)

if __name__ == '__main__':
    objName = Opinions_sentiment()
    objName.run_opinions_sentiment()