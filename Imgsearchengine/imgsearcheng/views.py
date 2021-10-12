import pandas
import re
import matplotlib
matplotlib.use('agg')
import statistics
from functools import reduce
import algo as algo
import numpy
import numpy as np
import random
from scipy import spatial
import self as self
from django.db.models import Q
from django.http import HttpResponseRedirect
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import distance
from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB
from django.shortcuts import render, render_to_response
from .models import Imagestore
import networkx as nx
import networkx.algorithms.tree.mst as ms
import matplotlib.pyplot as plt
import scipy as sc
# Create your views here.
import xml.etree.ElementTree as ET
from django.views.generic import DetailView
from .forms import Imagsearchform, Imagdetailform, Imagegraphform
from haystack import *
from haystack.query import SearchQuerySet
from scipy.sparse.csgraph import minimum_spanning_tree


from elasticsearch import Elasticsearch
from scipy.spatial.distance import euclidean,cdist
from scipy import stats
import math
from networkx.algorithms import tree

def images_listlview(request, slug):
    slug = request.POST.get('slug')
    slug = slug.lower()
    es = Elasticsearch()
    res = es.search(index="images",body={
        "query": {
            "multi_match" : {
                "query" : slug,
                "fields" : ["name",
        "tags",
        "category"]
            }
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            }
        ]
    })
    res =res['hits']
   
    res = res['hits']
    res = list(map(lambda x: x['_source'],res))
   
    strings = list(map(objToStringMapper,res))
  

    jaccards = []
    euclideans = []

    textualGraph = nx.Graph()
    visualGraph = nx.Graph()

    for index,obj in enumerate(res):
        node1 = obj['name']
        string1 = objToStringMapper(obj)

        cedValue1 = obj['ceddValue']
        for y in range(index + 1,len(res)):
            node2 = res[y]['name']
            string2 = objToStringMapper(res[y])
            ceddValue2 = res[y]['ceddValue']
            jaccard = round(distance.jaccard(string1,string2,),3)
            euclidean = round(getEuclideanDistance(cedValue1,ceddValue2),3)

            euclideans.append([euclidean,node1,node2])
            jaccards.append([jaccard,node1,node2])

            visualGraph.add_node(node1,alias='')
            visualGraph.add_node(node2,alias='')
            visualGraph.add_edge(node1,node2,label=euclidean)

            textualGraph.add_node(node1, alias='')
            textualGraph.add_node(node2,alias='')
            textualGraph.add_edge(node1,node2,label=jaccard)

    pos = nx.spring_layout(textualGraph) 
    nx.draw_networkx(textualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(textualGraph, 'alias')
    nx.draw_networkx_labels(textualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(textualGraph, 'label')
    nx.draw_networkx_edge_labels(textualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/initial_jaccard.jpg')
    plt.clf()

    pos = nx.spring_layout(visualGraph) 
    nx.draw_networkx(visualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(visualGraph, 'alias')
    nx.draw_networkx_labels(visualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(visualGraph, 'label')
    nx.draw_networkx_edge_labels(visualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/initial_euclidean.jpg')
    plt.clf()

    import copy

    beforeZahnGraph = copy.deepcopy(textualGraph)





    mappedJaccards = list(map(lambda x: x[0],jaccards))
    standardDeviation = statistics.stdev(mappedJaccards)
    mean = statistics.mean(mappedJaccards)
    threshold = standardDeviation + mean

    filteredJaccardArray = []
    for x in jaccards:
        jaccardVal = x[0]
        if(jaccardVal > threshold):
            filteredJaccardArray.append(x)
    textualGraph = nx.Graph()
    for index,x in enumerate(filteredJaccardArray):
        textualGraph.add_node(x[1],alias=index+1)
        textualGraph.add_node(x[2],alias=index+1)

        textualGraph.add_edge(x[1],x[2], label=x[0])

    pos = nx.spring_layout(textualGraph) 
    nx.draw_networkx(textualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(textualGraph, 'alias')
    nx.draw_networkx_labels(textualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(textualGraph, 'label')
    nx.draw_networkx_edge_labels(textualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/threshold_jaccard.jpg')
    plt.clf()
















    mappedEuclideans = list(map(lambda x: x[0],euclideans))

    #zscore = stats.zscore(mappedEuclideans)
    print("before", mappedEuclideans)
    zscore = __normalize(np.array(mappedEuclideans))
    print("after", zscore)
    textualGraph = nx.Graph()
    for index,x in enumerate(euclideans):
        textualGraph.add_node(x[1],alias=index+1)
        textualGraph.add_node(x[2],alias=index+1)

        textualGraph.add_edge(x[1],x[2], label=zscore[index])

    pos = nx.spring_layout(textualGraph) 
    nx.draw_networkx(textualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(textualGraph, 'alias')
    nx.draw_networkx_labels(textualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(textualGraph, 'label')
    nx.draw_networkx_edge_labels(textualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/zscore_euclidean.jpg')
    plt.clf()

    standardDeviation = statistics.stdev(zscore)
    mean = statistics.mean(zscore)
    threshold = standardDeviation + mean

    filteredEuclideanArray = []
    for index,x in enumerate(euclideans):
        euclideanVal = zscore[index]
        if(euclideanVal > threshold):
            x[0] = zscore[index]
            filteredEuclideanArray.append(x)
    visualGraph = nx.Graph()
    for index,x in enumerate(filteredEuclideanArray):
        visualGraph.add_node(x[1],alias=index+1)
        visualGraph.add_node(x[2],alias=index+1)

        visualGraph.add_edge(x[1],x[2], label=x[0])

    pos = nx.spring_layout(visualGraph) 
    nx.draw_networkx(visualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(visualGraph, 'alias')
    nx.draw_networkx_labels(visualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(visualGraph, 'label')
    nx.draw_networkx_edge_labels(visualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/threshold_euclidean.jpg')
    plt.clf()






    visualGraph = nx.Graph()
    for index,x in enumerate(filteredEuclideanArray):
        x[0] = 1-x[0]
        visualGraph.add_node(x[1],alias=index+1)
        visualGraph.add_node(x[2],alias=index+1)

        visualGraph.add_edge(x[1],x[2], label=x[0])

    pos = nx.spring_layout(visualGraph) 
    nx.draw_networkx(visualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(visualGraph, 'alias')
    nx.draw_networkx_labels(visualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(visualGraph, 'label')
    nx.draw_networkx_edge_labels(visualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/Minus_1_euclidean.jpg')
    plt.clf()

    jaccardFinalArray = np.array(list(map(lambda x: x[0],filteredJaccardArray))) * 0.5
    euclideanFinalArray = np.array(list(map(lambda x: x[0],filteredEuclideanArray))) * 0.5

    fArray = jaccardFinalArray + euclideanFinalArray



    visualGraph = nx.Graph()
    for index,x in enumerate(filteredEuclideanArray):
        x[0] = fArray[index]
        visualGraph.add_node(x[1],alias=index+1)
        visualGraph.add_node(x[2],alias=index+1)

        visualGraph.add_edge(x[1],x[2], label=x[0])

    pos = nx.spring_layout(visualGraph) 
    nx.draw_networkx(visualGraph, pos, with_labels=True)
    node_labels = nx.get_node_attributes(visualGraph, 'alias')
    nx.draw_networkx_labels(visualGraph, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(visualGraph, 'label')
    nx.draw_networkx_edge_labels(visualGraph, pos, edge_labels=edge_labels)
    
    plt.savefig('media/Multiply_0.5_Add_jaccard_euclidean.jpg')
    plt.clf()
    

    mst = tree.maximum_spanning_edges(visualGraph, algorithm='kruskal', data=True,weight='label')
    l = list(mst)
    #print("MST",list(mst))
    finalGraph =  getInconsistentEdges(l)
    dg = -1
    clusterName = ''
    #print(finalGraph.nodes)
    for index,n in enumerate(list(finalGraph.nodes)):
        
        if len(list(finalGraph.neighbors(n))) > dg:
            print(finalGraph.nodes[n])
            dg = len(list(finalGraph.neighbors(n)))
            clusterName = n
    # descSort = sorted(list(mst),reverse=True)
    # print(list(descSort))
    # for index,x in enumerate(strings):
    #     for y in range(index + 1,len(strings)):
    #         jaccards.append(distance.jaccard(x,strings[y]))
    # print(jaccards)
    # print(len(jaccards))
    
    # print(standardDeviation)
    # print(mean)
    # print(threshold)
    # thresholdComparedArray = []
    # for jScore in jaccards:
    #     if(jScore > threshold):
    #         thresholdComparedArray.append(jScore)
    # print(thresholdComparedArray)
    resultArr = []
    # xmlFiles = glob.glob('**/01 Bird/*.xml', recursive=True)
    # for xmlFilePath in xmlFiles:
    #     try:
    #         data = getImageFromXml(os.path.normpath(
    #         os.path.abspath(os.curdir) + '/' + xmlFilePath), slug)
            
    #         if(data):
    #             resultArr.append(data)
    #     except:
    #         pass
    # print(resultArr,sep='\n')
  
    images = None
    maxRatio = None
    # if(len(resultArr) > 0):
    #     ratiosArray = list(map(lambda x: x['ratio'],resultArr))
    #     clusterName = resultArr[0]['cluster']
    #     maxRatio = max(ratiosArray)
    #     print("aaaaaa", resultArr)
    #     images = list(map(lambda x: {"imageUrl" : x['imageUrl'],"mediaName" : x['mediaName']},resultArr))
    # imageslist = SearchQuerySet().autocomplete(
    #     content_auto=request.POST.get('slug', ''))
    # allimages = Imagestore.objects.exclude(
    #     Q(img_category__iexact=slug) | Q(img_category__icontains=slug))
    # uservalue = request.POST.get('slug')
    # js = []
    # jsdict = {}
    # for ob in allimages:
    #     for obj2 in allimages:
    #         js.append(distance.jaccard(ob.img_usertag, obj2.img_usertag))
    #         keys = ob.img_usertag
    #         jsdict[keys] = distance.jaccard(ob.img_usertag, obj2.img_usertag)
    # jsmax = max(js)
    # jsmaxkey = max(jsdict, key=jsdict.get)
    # return render_to_response('imagelist.html',
    #                           {'imageslist': imageslist, 'uservalue': uservalue, 'allimages': allimages, 'jsmaxkey': jsmaxkey, 'jsmax': jsmax
    #                            })
    from networkx.readwrite import json_graph
    newRes = list(map(chuchu,res))
    import copy
    _newRes = copy.deepcopy(newRes)
    for x in _newRes:
        temp = []
        try:
            nodeName = x['name']
            neighbors = list(beforeZahnGraph.neighbors(nodeName))
            #print(nodeName,'neighbors', neighbors)
            for nbr in neighbors:
                
                for aaa in _newRes:
                    if aaa['name'] == nbr:
                        #cpy = copy.deepcopy(aaa)
                        temp.append(aaa)
                        
                        
                #x['neighbors']=list(filter(lambda poo: poo['name'] == nbr,_newRes))
            x['neighbors'] = copy.deepcopy(temp)
            #print("===>>", len(x['neighbors']))
        except:
            x['neighbors'] = temp
    import json
    # print('===>>>> ',json.dumps(_newRes))
    return render(request,'imagelist.html',{'images' : _newRes,'clusterName':clusterName,'maxRatio' : maxRatio, 'graphData' : json_graph.node_link_data(finalGraph)})

def chuchu(v):
    del v['ceddValue']
    return v

def getbase64(request):
    import base64
    import requests
    from django.http import HttpResponse
    data = base64.b64encode(requests.get(request.GET.get('url', '')).content).decode('utf-8')
    return HttpResponse("data:image/png;base64," + data)
def searchgraphview(request, slug=None):
    pass


freeText = None
imageUrl= None
tags= None
name= None
neighbors = None
def images_detailview(request):
 
    if request.method=='POST':
        global freeText,imageUrl,tags,name
        freeText = request.POST.get('freeText')
        imageUrl = request.POST.get('imageUrl')
        tags = request.POST.get('tags')
        name = request.POST.get('name')
        a = request.POST.get('neighbors')
        import urllib
        neighbors = urllib.parse.unquote(a)
        
        # imageobject = Imagestore.objects.get(pk=pk)
        # imageslist = Imagestore.objects.filter(Q(img_category__icontains=slug))
        import ast
        try:
            neighbors = ast.literal_eval(neighbors)
           
        except:
            
            pass
        
    args = {'freeText': freeText, 'imageUrl' : imageUrl, 'tags' : tags,'name' : name, 'neighbors' : neighbors}
    template_name = 'imagedetail.html'
    return render(request, template_name, args)



#######################################################################################
def getImageFromXml(xmlFilePath, slug):
    
    tags = []  # tags which will be extracted from xml file
    with open(xmlFilePath) as fd:
        doc = xmltodict.parse(fd.read())  # read xml file
        # xml file inside cursor navigation
        tagsArr = doc['RUCoD']['Header']['Tags']['MetaTag']

        for tag in tagsArr:
            tags.append(str(tag['#text']))  # append extracted tag

        # map tags to lowercase
        tags = list(map(lambda x: x.lower(), tags))

        # map tags to remove white space and numeric characters using CustomMapper function
        tags = list(map(customMapper, tags))

        matchRatios = []
        for t in tags:
            # get 'slug' and 'tag' match score ratio
            # append ratio in matchRatios
            matchRatios.append(float(getStringMatchRatio(t, slug)))

        # max ratio in matchRatios list
        if(max(matchRatios) >= 0.8): # set minimum threshold to match is 0.5, max is 1
           
            # return image url
            return {
                "ratio" : max(matchRatios),
                "imageUrl": doc['RUCoD']['Header']['ContentObjectTypes']['MultimediaContent']['MediaLocator']['MediaUri'],
                "cluster" : tags[-1].upper(),
                "mediaName": doc['RUCoD']['Header']['ContentObjectTypes']['MultimediaContent']['MediaName']
            }
        else:
            # return None as no tag matched
            return None


"""Custom Mapper function to remove white space and remove numeric characters"""


def customMapper(string):
    splt = string.split()
    if len(splt) > 1:
        return splt[1]
    else:
        return splt[0]


"""Get 2 Strings match ratio"""


def getStringMatchRatio(string1, string2):
    return SequenceMatcher(None, string1, string2).ratio()



def objToStringMapper(obj):
    text = ''
    s = ","
    for key in obj.keys():
      
        if key == 'tags':
           
            text = text + ' ' + s.join(obj[key])
        elif key == 'ceddValue':
            pass
        else:
            text = text + ' ' + obj[key]
    return text

def getEuclideanDistance(x,y):
    # V = np.add(np.array(x),np.array(y))
    # V = np.sqrt(V)
    # # dist = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(x, y)]))
    # dist = seuclidean(x,y,V)
    dist = euclidean(x,y)
    return dist

def getInconsistentEdges(arrayOfEdges):

    G = nx.Graph()
    for obj in arrayOfEdges:
        print(obj[0],obj[1],obj[2]['label'])
        G.add_node(obj[0],alias="")
        G.add_node(obj[1],alias="")
        G.add_edge(obj[0],obj[1],weight=obj[2]['label'])
    
    # pos = nx.spring_layout(G) 
    # nx.draw_networkx(G, pos, with_labels=True)
    # node_labels = nx.get_node_attributes(G, 'alias')
    # nx.draw_networkx_labels(G, pos, node_labels)
    
    # # edge labels
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # plt.savefig('media/ulti.jpg')
    # plt.clf()
    graphs = []


    for node in G.nodes:
       graphs.append(adjacencyFunction(G,node))
    pos = nx.spring_layout(G) 
    nx.draw_networkx(G, pos, with_labels=True)
    node_labels = nx.get_node_attributes(G, 'alias')
    nx.draw_networkx_labels(G, pos, node_labels)
    
    # edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.savefig('media/MST.jpg')
    plt.clf()
    return G

def adjacencyFunction(G,node):
    print("\n\n\n","adjacent edges to " + node,list(G.neighbors(node)),"\n")

    adjacentEdges = list(G.neighbors(node))
    if len(adjacentEdges) == 0:
        return
    node1 = node
    node2 = adjacentEdges[0]
    currentEdgeWeight = G.get_edge_data(node1,node2)['weight']

    adjacentEdgesWeights = []
    node1AdjacentEdges = list(G.neighbors(node1))
    node1AdjacentEdges = list(filter(lambda x: x != node1 and x!=node2,node1AdjacentEdges))
    node2AdjacentEdges = list(G.neighbors(node2))
    node2AdjacentEdges = list(filter(lambda x: x != node2 and x!=node1,node2AdjacentEdges))
   
    if len(node1AdjacentEdges) != 1:
        for x in node1AdjacentEdges:
            y = G.get_edge_data(node1,x)['weight']
            adjacentEdgesWeights.append(
            y   
            )
            
    if len(node2AdjacentEdges) !=1:

        for x in node2AdjacentEdges:
            y = G.get_edge_data(node2,x)['weight']
            adjacentEdgesWeights.append(
            y   
            )
            
    
    
    try:
        avg = sum(adjacentEdgesWeights) / len(adjacentEdgesWeights)
        if currentEdgeWeight < avg:
            #remove the edge
            G.remove_edge(node1,node2)  
    except:
        pass
    
    # if len(adjacentEdges) == 1:
    #     edgeWeight = G.get_edge_data(node,adjacentEdges[0])['weight']
    #     print("currently taken",node,adjacentEdges[0],edgeWeight)
    #     avg = (0) / 2
    #     print("weight less than avg",edgeWeight < avg)
    #     if edgeWeight < avg:
    #         print(node,adjacentEdges[0],"edge removed")
    #         G.remove_edge(node,adjacentEdges[0])
           
    # else:
    #     edgeWeight = G.get_edge_data(node,adjacentEdges[0])['weight']
    #     print("currently taken",node,adjacentEdges[0],edgeWeight)
    #     weights = []
    #     for index,adjacentEdge in enumerate(adjacentEdges):
    #         if index == 0:
    #             pass
    #         else:
    #             print("adjacent taken",node,adjacentEdge,G.get_edge_data(node,adjacentEdge)['weight'])
    #             weights.append(G.get_edge_data(node,adjacentEdge)['weight'])
    #     print("weights", weights)
    #     avg = (sum(weights)) / (len(weights))
    #     print("average", avg)
    #     print("weight less than avg",edgeWeight < avg)
    #     if edgeWeight < avg:
    #         print("edge removed")
    #         G.remove_edge(node,adjacentEdges[0])
    return None

def __normalize( data ) :
    # Save the Real shape of the Given Data
    shape = data.shape
    # Smoothing the  Given Data Valuesto 1 dimension
    data = np.reshape( data , (-1 , ) )
    # Find MinValue and MaxValue
    MaxValue = np.max( data )
    MinValue = np.min( data )
    # Normalized values are store in a newly created array
    normalized_values = list()
    # Iterate through every value in data
    for AttributeValue in data:
    # Normalize
        AttributeValue_normalized = ((AttributeValue) - (MinValue))/((MaxValue)-(MinValue))
        # Append it in the array
        normalized_values.append(AttributeValue_normalized)
    # Convert to numpy array
    n_array = np.array( normalized_values )
    # Reshape the array to its Real shape and return it.
    return np.reshape( n_array , shape )