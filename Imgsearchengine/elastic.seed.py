from elasticsearch import Elasticsearch
import os
import glob
import xmltodict


def getAllXmlFiles():
    xmlFiles = glob.glob('./DATASET_FINAL_WORK/**/*.xml', recursive=True)

    return [os.path.normpath(
        os.path.abspath(os.curdir) + '/' + xmlFilePath) for xmlFilePath in xmlFiles]


def parseXmlData(xmlPath):
    obj = {

    }
    with open(xmlPath) as fd:
        doc = xmltodict.parse(fd.read())
        obj['name'] = doc['RUCoD']['Description']['L_Descriptor']['MediaName']
        obj["imageUrl"] = doc['RUCoD']['Header']['ContentObjectTypes']['MultimediaContent']['MediaLocator']['MediaUri']
        obj['freeText'] = doc['RUCoD']['Header']['ContentObjectTypes']['MultimediaContent']['FreeText']

        tagsArr = doc['RUCoD']['Header']['Tags']['MetaTag']
        tags = []
        for tag in tagsArr:
            tags.append(str(tag['#text']))
        obj['tags'] = tags
        obj['category'] = tags[-1]
    return obj


def postToElasticSearch(esInstance, data, id):
    try:
        res = esInstance.index(
            index="images", doc_type="image", id=id, body=data)
        return res
    except:
        pass


def getCeddFileData(fileNumber):
    with open('./CED_IMAGES/' + fileNumber + '.txt', mode='r') as f:
        arr = []
        fileData = f.read()
        splt = fileData.split(' ')
        return list(map(lambda x: int(x), splt))


def getCeddData(xmlPath, ceddPathsArray):
    sep = '_____'
    normalPath = os.path.normpath(xmlPath)
    normalPath = normalPath.replace('\\', sep)
    splitPath = normalPath.split(sep)
    fileName = splitPath[-1]

    for x in ceddPathsArray:
        cedd_normalPath = os.path.normpath(x)
        cedd_normalPath = cedd_normalPath.replace('\\', sep)
        cedd_splitPath = cedd_normalPath.split(sep)
        cedd_fileName = cedd_splitPath[-1]
        temp = cedd_fileName.split(' ')

        if(fileName == temp[0]):

            return getCeddFileData(temp[1])


if __name__ == "__main__":
    cedPaths = open('CEDD_PATHS.txt', mode='r')
    cedPaths = cedPaths.read()
    cedPaths = cedPaths.split('\n')

    # print(cedPaths)
    # print(len(cedPaths))
    xmlFiles = getAllXmlFiles()
    # print(xmlFiles)

    temp = []

    for x in xmlFiles:

        try:
            temp.append([parseXmlData(x), getCeddData(x, cedPaths)])
        except:
            pass
    print(temp, sep="\n\n")
    print(len(temp))
    try:

        es = Elasticsearch()
        for index, data in enumerate(temp):
            try:
                print("About to process")
                temp = data[0]
                temp['ceddValue'] = data[1]
                print(postToElasticSearch(es, temp, index+1), sep="\n")
            except:
                pass
    except:
        pass
