from Trie import Trie
from TrieNode import TrieNode


class StringSearch():
    def __init__(self):
        self._first = {}
        self._keywords = []
    
    def SetKeywords(self,keywords):
        self._keywords = keywords
        root = TrieNode()
        allNodeLayer = {}

        for i in range(len(self._keywords)): 
            p = self._keywords[i]
            nd = root
            for j in range(len(p)):
                nd = nd.Add(ord(p[j]))
                if (nd.Layer == 0):
                    nd.Layer = j + 1
                    if nd.Layer in allNodeLayer:
                        allNodeLayer[nd.Layer].append(nd)
                    else:
                        allNodeLayer[nd.Layer]=[]
                        allNodeLayer[nd.Layer].append(nd)
            nd.SetResults(i)


        allNode = []
        allNode.append(root)
        for key in allNodeLayer.keys():
            for nd in allNodeLayer[key]:
                allNode.append(nd)
        allNodeLayer = None

        for i in range(len(allNode)): 
            if i == 0:
                continue
            nd=allNode[i]
            nd.Index = i
            r = nd.Parent.Failure
            c = nd.Char
            while (r != None and (c in r.m_values)==False):
                r = r.Failure
            if (r == None):
                nd.Failure = root
            else:
                nd.Failure = r.m_values[c]
                for key2 in nd.Failure.Results :
                    nd.SetResults(key2)
        root.Failure = root

        allNode2 = []
        for i in range(len(allNode)): 
            allNode2.append(Trie())
        
        for i in range(len(allNode2)): 
            oldNode = allNode[i]
            newNode = allNode2[i]

            for key in oldNode.m_values :
                index = oldNode.m_values[key].Index
                newNode.Add(key, allNode2[index])
            
            for index in range(len(oldNode.Results)):
                item = oldNode.Results[index]
                newNode.SetResults(item)
            

            oldNode=oldNode.Failure
            while oldNode != root:
                for key in oldNode.m_values :
                    if (newNode.HasKey(key) == False):
                        index = oldNode.m_values[key].Index
                        newNode.Add(key, allNode2[index])
                for index in range(len(oldNode.Results)): 
                    item = oldNode.Results[index]
                    newNode.SetResults(item)
                oldNode=oldNode.Failure
        allNode = None
        root = None

        self._first = allNode2[0]
    

    def FindFirst(self,text):
        ptr = None
        for index in range(len(text)):
            t =ord(text[index]) 
            tn = None
            if (ptr == None):
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if (tn==None):
                    tn = self._first.TryGetValue(t)
                
            if (tn != None):
                if (tn.End):
                    return self._keywords[tn.Results[0]]
            ptr = tn
        return None

    def FindAll(self,text):
        ptr = None
        list = []

        for index in range(len(text)): 
            t =ord(text[index]) 
            tn = None
            if (ptr == None):
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if (tn==None):
                    tn = self._first.TryGetValue(t)
                
            
            if (tn != None):
                if (tn.End):
                    for j in range(len(tn.Results)):
                        item = tn.Results[j]
                        list.append(self._keywords[item])
            ptr = tn
        return list


    def ContainsAny(self,text):
        ptr = None
        for index in range(len(text)): 
            t =ord(text[index]) 
            tn = None
            if (ptr == None):
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if (tn==None):
                    tn = self._first.TryGetValue(t)
            
            if (tn != None):
                if (tn.End):
                    return True
            ptr = tn
        return False
    
    def Replace(self,text, replaceChar = '*'):
        result = list(text) 

        ptr = None
        for i in range(len(text)): 
            t =ord(text[i]) 
            tn = None
            if (ptr == None):
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if (tn==None):
                    tn = self._first.TryGetValue(t)
            
            if (tn != None):
                if (tn.End):
                    maxLength = len(self._keywords[tn.Results[0]])
                    start = i + 1 - maxLength
                    for j in range(start,i+1): 
                        result[j] = replaceChar
            ptr = tn
        return ''.join(result) 
