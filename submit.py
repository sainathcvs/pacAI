from __future__ import division
import random
import pickle as pkl
import argparse
import csv
import numpy as np
import scipy.stats


'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274

#A random tree construction for illustration, do not use this in your code!
def create_random_tree(depth):
    if(depth >= 7):
        if(random.randint(0,1)==0):
            return TreeNode('T',[])
        else:
            return TreeNode('F',[])

    feat = random.randint(0,273)
    root = TreeNode(data=str(feat))

    for i in range(5):
        root.nodes[i] = create_random_tree(depth+1)

    return root

def entropy(list):
	"""Calculates entropy for given 1D list"""
	val,counts = np.unique(list,return_counts=True)
	probability = counts/len(list)
	entr = np.dot(-1*probability, np.log2(probability))
#	print("For lsit:", list, " entropy is ",entr)
	return entr

"""def infogain(label_list,classes):
	#Takes 1D occurence list for a given attribute, stacks with label list, computes infogain
	data_l = np.reshape(label_list,(len(label_list),1))
	data_c = np.reshape(classes,(len(classes),1))
	data = np.hstack((data_l,data_c))
	val,counts = np.unique(label_list)
	prob = val/counts
	entrpy = []
	for v in val:
		entrpy.append(entropy( data[np.where(data[:,0] == v)] ))
	entrpy = np.array(entrpy)
	print("Entropy list:",entrpy)
	info = np.dot(prob,entrpy)"""

def infogain(label_list,classes):
	"""Takes 1D occurence list for a given attribute, stacks with label list, computes infogain"""
	data_l = np.reshape(label_list,(len(label_list),1))
	data_c = np.reshape(classes,(len(classes),1))
	data = np.hstack((data_l,data_c))
	val,counts = np.unique(label_list, return_counts=True)
	prob = counts/len(label_list)
	entrpy = []
	for v in val:
		entrpy.append(entropy( data[np.where(data[:,0] == v)][:,1] ))
	entrpy = np.array(entrpy)
#	print("Entropy list:",entrpy)
	info = np.dot(prob,entrpy)
	return entropy(classes) - info

def chiSquareCriterion(data, threshold):
    N = 0
    ncount = 0
    pcount = 0
    attr_count = []
    n=[]
    p=[]
    for list in data:
        N += len(list)
        val, count = np.unique(list, return_counts=True)
        if np.array_equal(val,[0,1]):
            ncount += count[0]
            pcount += count[1]
            n.append(count[0])
            p.append(count[1])
        elif np.array_equal(val,[1]):
            pcount += count[0]
            p.append(count[0])
            n.append(0)
        elif np.array_equal(val,[0]):
            ncount += count[0]
            n.append(count[0])
            p.append(0)
        attr_count.append(len(list))
    attr_count = np.array(attr_count)
    p_ = attr_count * (pcount/N)
    n_ = attr_count * (ncount/N)
    S=0
    for index in range(0,len(p_)):
       S += (p_[index] - p[index])*(p_[index] - p[index])/p_[index] + (n_[index] - n[index])*(n_[index] - n[index])/n_[index]
    #Adding code to calculate p value
    pValue = 1 - scipy.stats.chi2.cdf(S, len(data)-1)
    return pValue<threshold

def tree_traversal(root,test_sample):
    if root == None:
        return -1
    if root.data == 'T':
#        print "true ", root.data
        return 1
    if root.data == 'F':
#        print "false ", root.data
        return 0
    idx = root.data
    tv = test_sample[idx]
    return tree_traversal(root.nodes[tv-1],test_sample)




def buildTree(node, data, pval, indices):
    #send 2D numpy array with results also appended
    #exit conditions
    if len(data) == 0:
        return
    if len(data[0]) == 0:
        v, c = np.unique(data[:,-1])
        m_i = np.argmax(c)
        if v[m_i] == 0:
            res = 'F'
        else:
            res = 'T'
        node.data = res
        return node
    if len(np.unique(data[:,-1])) == 1:
        if np.unique(data[:,-1])[0] == 0:
            res = 'F'
        else:
            res = 'T'
        node.data = res
        return node

    info_g = []
    for v in range(0,len(data[0])-1):
        info_g.append(infogain(data[:,v], data[:,-1]))
    max_index = info_g.index(max(info_g))
    node.data = indices[max_index]
    chi_data = []
    for i in range(1,6):
        d=data[np.where(data[:, max_index] == i)]
        _d = d[:,-1]
        if len(_d) != 0:
            chi_data.append( _d )

    if chiSquareCriterion(chi_data, pval) and info_g[max_index] > 0:
        #build children, make recursive call
#        print("Expanding further.")
        for i in range(0,5):
            newNode = TreeNode()
            node.nodes[i] = newNode
            d = data[ np.where(data[:,max_index] == i+1) ]
            subData = np.hstack((d[:,:max_index],d[:,max_index+1:]))
            subIndices = indices[:max_index] + indices[max_index+1:]
            #print("max_index",max_index,"info gain:",info_g[max_index])
            #if(max_index == 0):
                #print data
            buildTree(newNode, subData, pval,subIndices)
    else:
        #simply build leaf nodes
#        print("Chisq criterion failed. Creating elaf nodes")
        for i in range(0,5):
            if len(chi_data) == 0 or i >= len(chi_data):
                newNode = TreeNode(data='F')
                node.nodes[i] = newNode
                continue;
            v,c = np.unique(chi_data[i], return_counts=True)
            m_i = np.argmax(c)
            if v[m_i] == 0:
                newNode = TreeNode(data='F')
            else:
                newNode = TreeNode(data='T')
            node.nodes[i] = newNode

    return node


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")
#s = create_random_tree(0)
newNode = TreeNode()
x_train = np.array(Xtrain)
y_train = np.array(Ytrain).reshape(len(Ytrain),1)
data = np.hstack((x_train, y_train))
indc = range(0,x_train.shape[1])
s = buildTree(newNode, data, pval,indc)

s.save_tree(tree_name)
print("Testing...")
Ypredict = []
#generate random labels
for i in range(0,len(Xtest)):
	Ypredict.append([np.random.randint(0,2)])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")

#mine
Ytest = []
ftest_label = Xtest_name.split('.')[0] + '_label.csv'
with open(ftest_label, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        rw = int(row[0])
        Ytest.append(rw)
Ypredict = []
count = 0
#predict labels for test samples
for i in range(0,len(Xtest)):
    Ypredict.append(tree_traversal(newNode,Xtest[i]))
    if Ytest[i]==Ypredict[i]:
        count+=1

"""for i in range(0,len(Xtrain)):
    Ypredict.append(tree_traversal(newNode,Xtrain[i]))
    if Ytrain[i]==Ypredict[i]:
        count+=1"""

#pred = tree_traversal(newNode)
print count
print "accuracy is ",count/len(Ytest)






