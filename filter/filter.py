import json
import re
from stringSearch import StringSearch

def filterText(input_text):# bool
    ret = {}
    try :
        isFiltered = search.ContainsAny(input_text)
        allBadWordList = search.FindAll(input_text)
        replaceText = search.Replace(input_text)
        allBadWordList = list(set(allBadWordList))#去重

        positionList = []
        for badWord in allBadWordList:
            w = [i.span() for i in re.finditer(badWord, input_text)]
            for ele in w :
                t = str(ele[0]) + '-' + str(ele[1])
                positionList.append(t)
         
        positionString = ','.join(positionList)
        badWordString = ','.join(allBadWordList)

        if isFiltered :
            ret['code'] = 0
            ret['riskLevel'] = 'REJECT'
            ret['postion'] = positionString
            ret['badWord'] = badWordString
            ret['replaceText'] = replaceText 

        return json.dumps(ret)

    except Exception as e:  
        ret['code'] = 1
        ret['message'] = 'ERROR! Please try again!'
        ret['exceptionMessage'] = e
        return json.dumps(ret)

def replaceText(input_text):
    ret = {}
    try :
        isFiltered = search.ContainsAny(input_text)
        replaceText = search.Replace(input_text)  
        if isFiltered :
            ret['code'] = 0
            ret['riskLevel'] = 'REJECT'
            ret['replaceText'] = replaceText 
        return json.dumps(ret)

    except Exception as e:  
        ret['code'] = 1
        ret['message'] = 'ERROR! Please try again!'
        ret['exceptionMessage'] = e
        return json.dumps(ret)

if __name__ == '__main__' : 
    with open('./dict/key.txt', 'r', encoding = 'utf-8') as f :
        plain_text = f.read()
        search = StringSearch()
        search.SetKeywords(plain_text.split('|'))

        input_text = input()
        print(filterText(input_text))
        
        print(replaceText(input_text))
