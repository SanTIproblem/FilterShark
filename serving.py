import numpy as np
from flask import Flask, request, render_template
import pickle
import json
import re
from filter.stringSearch  import StringSearch
from dl_filter import predict

app = Flask(__name__)  # 初始化APP


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predictone", methods=["POST"])
def predictone():
    int_features = request.form.get("DL")
    prediction = predict(int_features)

    output = prediction[0]

    judge = ["Accept", "Reject"]

    return render_template(
        "index.html", prediction_text="word judge: {}".format(judge[output])
    )


def filterText(input_text):  # bool
    ret = {}
    try:
        isFiltered = search.ContainsAny(input_text)
        allBadWordList = search.FindAll(input_text)
        replaceText = search.Replace(input_text)
        allBadWordList = list(set(allBadWordList))  # 去重

        positionList = []
        for badWord in allBadWordList:
            w = [i.span() for i in re.finditer(badWord, input_text)]
            for ele in w:
                t = str(ele[0]) + '-' + str(ele[1])
                positionList.append(t)

        positionString = ','.join(positionList)
        badWordString = ','.join(allBadWordList)

        if isFiltered:
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
    try:
        isFiltered = search.ContainsAny(input_text)
        replaceText = search.Replace(input_text)
        if isFiltered:
            ret['code'] = 0
            ret['riskLevel'] = 'REJECT'
            ret['replaceText'] = replaceText
        return json.dumps(ret)

    except Exception as e:
        ret['code'] = 1
        ret['message'] = 'ERROR! Please try again!'
        ret['exceptionMessage'] = e
        return json.dumps(ret)


@app.route("/predicttwo", methods=["POST"])
def predicttwo():
    input_text = request.form.get("nonDL")
    f_text = filterText(input_text)
    return render_template("index.html", prediction_text="word judge: {}".format(f_text))


if __name__ == "__main__":
    with open('./filter/dict/key.txt', 'r', encoding='utf-8') as f:
        plain_text = f.read()
        search = StringSearch()
        search.SetKeywords(plain_text.split('|'))

        input_text = input()
        print(filterText(input_text))

        print(replaceText(input_text))

    app.run(debug=True)
