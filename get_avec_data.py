import numpy as np
np.random.seed(1234)
#
#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
#import torch.optim as optim
#
#import argparse
#import time
#
#from sklearn.metrics import mean_absolute_error
#from scipy.stats import pearsonr
#
#import pandas as pd
#
#from model import AVECModel, MaskedMSELoss
##from dataloader import AVECDataset
#from torch.utils.data import Dataset
import pickle
#from torch.nn.utils.rnn import pad_sequence

path = './DialogueRNN_features/DialogueRNN_features/AVEC_features/AVEC_features_{}.pkl'.format(1)
videoIDs, videoSpeakers, videoLabels, videoText,\
            videoAudio, videoVisual, videoSentence,\
            trainVid, testVid = pickle.load(open(path, 'rb'),encoding='latin1')
            
train_dic = {}
for vid in trainVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

num = int(len(trainVid)*0.9)
i = 0
train_dic2 = {}
val_dic = {}
for vid in train_dic.keys():
    if i < num:
        train_dic2[vid] = train_dic[vid]
        i += 1
    else:
        val_dic[vid] = train_dic[vid]
        
    
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_train_attr1.p", 'wb')
pickle.dump(train_dic2, f)
f.close()  
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_val_attr1.p", 'wb')
pickle.dump(val_dic, f)
f.close()  

train_dic = {}
for vid in testVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

f = open("/home/dzhang/IJCAI/2020/avec/avec_features_test_attr1.p", 'wb')
pickle.dump(train_dic, f)
f.close() 

path = './DialogueRNN_features/DialogueRNN_features/AVEC_features/AVEC_features_{}.pkl'.format(2)
videoIDs, videoSpeakers, videoLabels, videoText,\
            videoAudio, videoVisual, videoSentence,\
            trainVid, testVid = pickle.load(open(path, 'rb'),encoding='latin1')
            
train_dic = {}
for vid in trainVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
        sample = {}
        if i == 0:
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

num = int(len(trainVid)*0.9)
i = 0
train_dic2 = {}
val_dic = {}
for vid in train_dic.keys():
    if i < num:
        train_dic2[vid] = train_dic[vid]
        i += 1
    else:
        val_dic[vid] = train_dic[vid]
        
    
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_train_attr2.p", 'wb')
pickle.dump(train_dic2, f)
f.close()  
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_val_attr2.p", 'wb')
pickle.dump(val_dic, f)
f.close()  

train_dic = {}
for vid in testVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

f = open("/home/dzhang/IJCAI/2020/avec/avec_features_test_attr2.p", 'wb')
pickle.dump(train_dic, f)
f.close() 

path = './DialogueRNN_features/DialogueRNN_features/AVEC_features/AVEC_features_{}.pkl'.format(3)
videoIDs, videoSpeakers, videoLabels, videoText,\
            videoAudio, videoVisual, videoSentence,\
            trainVid, testVid = pickle.load(open(path, 'rb'),encoding='latin1')
            
train_dic = {}
for vid in trainVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

num = int(len(trainVid)*0.9)
i = 0
train_dic2 = {}
val_dic = {}
for vid in train_dic.keys():
    if i < num:
        train_dic2[vid] = train_dic[vid]
        i += 1
    else:
        val_dic[vid] = train_dic[vid]
        
    
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_train_attr3.p", 'wb')
pickle.dump(train_dic2, f)
f.close()  
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_val_attr3.p", 'wb')
pickle.dump(val_dic, f)
f.close()  

train_dic = {}
for vid in testVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

f = open("/home/dzhang/IJCAI/2020/avec/avec_features_test_attr3.p", 'wb')
pickle.dump(train_dic, f)
f.close() 

path = './DialogueRNN_features/DialogueRNN_features/AVEC_features/AVEC_features_{}.pkl'.format(4)
videoIDs, videoSpeakers, videoLabels, videoText,\
            videoAudio, videoVisual, videoSentence,\
            trainVid, testVid = pickle.load(open(path, 'rb'),encoding='latin1')
            
train_dic = {}
for vid in trainVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

num = int(len(trainVid)*0.9)
i = 0
train_dic2 = {}
val_dic = {}
for vid in train_dic.keys():
    if i < num:
        train_dic2[vid] = train_dic[vid]
        i += 1
    else:
        val_dic[vid] = train_dic[vid]
        
    
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_train_attr4.p", 'wb')
pickle.dump(train_dic2, f)
f.close()  
f = open("/home/dzhang/IJCAI/2020/avec/avec_features_val_attr4.p", 'wb')
pickle.dump(val_dic, f)
f.close()  

train_dic = {}
for vid in testVid:
    train_dic[vid] = []
    not_copy = None
    # copy_no = 0
    for i in range(len(videoText[vid])):
        source_index = i
        target_index = None
        speaker_anchor = videoSpeakers[vid][i]
        change_anchor = True
        for j in range(i+1,len(videoText[vid])):
            if (videoSpeakers[vid][j] == speaker_anchor)&change_anchor:
                source_index = j
                continue
            change_anchor = False 
            if (videoSpeakers[vid][j] == speaker_anchor)&(not change_anchor):
                target_index = j
                break

        if not_copy == videoSentence[vid][source_index]: 
            continue
        
        if target_index == None:
            break
        
#        left_context_labels = []
#        for label in videoLabels[vid][:source_index]:
#            left_context_labels.append(str(label))
#            
#        right_context_labels = []       
#        for label in videoLabels[vid][source_index+1:target_index]:
#            right_context_labels.append(str(label))
        
#        sample = {'left context text':[],'left context emotions':[],'left context speakers','source utterance text','source utterance emotion','source utterance speaker','right context text','right context emotions','right context speakers','target utterance text','target utterance emotion','target utterance speaker','change'])
        sample = {}
        if i == 0:
#            csv_writer.writerow(['','','',videoSentence[vid][source_index],str(videoLabels[vid][source_index]),videoSpeakers[vid][source_index],'<SSS>'.join(videoSentence[vid][source_index+1:target_index]),\
#                                 '<SSS>'.join(right_context_labels), '<SSS>'.join(videoSpeakers[vid][source_index+1:target_index]),videoSentence[vid][target_index], str(videoLabels[vid][target_index]), videoSpeakers[vid][target_index], change_state])
            sample['left context text'] = []
            sample['left context emotions'] = []
            sample['left context speakers'] = []

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        else:            
            sample['left context text'] = videoSentence[vid][:source_index]
            sample['left context emotions'] = videoLabels[vid][:source_index]
            sample['left context speakers'] = videoSpeakers[vid][:source_index]
#            sample['left context textual features'] = videoText[vid][:source_index]
#            sample['left context acoustic features'] = videoAudio[vid][:source_index]

            sample['source utterance text'] = videoSentence[vid][source_index]
            sample['source utterance emotion'] = videoLabels[vid][source_index]
            sample['source utterance speaker'] = videoSpeakers[vid][source_index]
#            sample['source utterance textual features'] = videoText[vid][source_index]
#            sample['source utterance acoustic features'] = videoAudio[vid][source_index]

            sample['right context text'] = videoSentence[vid][source_index+1:target_index]
            sample['right context emotions'] = videoLabels[vid][source_index+1:target_index]
            sample['right context speakers'] = videoSpeakers[vid][source_index+1:target_index]
#            sample['right context textual features'] = videoText[vid][source_index+1:target_index]
#            sample['right context acoustic features'] = videoAudio[vid][source_index+1:target_index]
            
            sample['change state'] = videoLabels[vid][target_index] - videoLabels[vid][source_index]
        
        train_dic[vid].append(sample)
        not_copy = videoSentence[vid][source_index]

f = open("/home/dzhang/IJCAI/2020/avec/avec_features_test_attr4.p", 'wb')
pickle.dump(train_dic, f)
f.close() 