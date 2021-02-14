#imports
from flask import Flask, render_template, request
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
import nltk
# nltk.download('punkt')
# nltk.download()
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import os

import requests
import pandas
import csv
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import speech_recognition as sr 
import pyttsx3 

import pytesseract

from PIL import Image
from googletrans import Translator

r = sr.Recognizer() 

app = Flask(__name__)
# #create chatbot
# englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(englishBot)
# trainer.train("chatterbot.corpus.english") #train the chatter bot for english
# #define app routes


with open("intents.json") as file:
		data = json.load(file)

with open("pintents.json") as file2:
		data2 = json.load(file2)



# try:
# 	with open("data.pickle", "rb") as f:
# 		words, labels, training, output = pickle.load(f)
# except:
words = []
labels = []
docs_x = []
docs_y = []


words2 = []
labels2 = []
docs_x2 = []
docs_y2 = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

for intent2 in data["intents"]:
	for pattern2 in intent2["patterns"]:
		wrds2 = nltk.word_tokenize(pattern2)
		words2.extend(wrds2)
		docs_x2.append(wrds2)
		docs_y2.append(intent2["tag"])

	if intent2["tag"] not in labels2:
		labels2.append(intent2["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

words2 = [stemmer.stem(w2.lower()) for w2 in words2 if w2 != "?"]
words2 = sorted(list(set(words2)))

labels = sorted(labels)

labels2 = sorted(labels2)

training = []
output =[]

training2 = []
output2 =[]

out_empty = [0 for _ in range(len(labels))]

out_empty2 = [0 for _ in range(len(labels2))]

for x, doc in enumerate(docs_x):
	bag = []

	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


for x2, doc2 in enumerate(docs_x):
	bag2 = []

	wrds2 = [stemmer.stem(w2) for w2 in doc2]

	for w2 in words2:
		if w2 in wrds2:
			bag2.append(1)
		else:
			bag2.append(0)

	output_row2 = out_empty2[:]
	output_row2[labels2.index(docs_y2[x2])] = 1

	training2.append(bag2)
	output2.append(output_row2)

training2 = numpy.array(training2)
output2 = numpy.array(output2)

	# with open("data.pickle", "wb") as f:
	# 	pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
# 	model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=1, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s,cwords):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def bag_of_words2(s2,cwords2):
	bag2 = [0 for _ in range(len(words))]

	s_words2 = nltk.word_tokenize(s)
	s_words2 = [stemmer.stem(word2.lower()) for word2 in s_words2]

	for se2 in s_words2:
		for i2, w2 in enumerate(words2):
			if w2 == se2:
				bag2[i] = 1

	return numpy.array(bag2)


def SpeakText(command): 
	
	# Initialize the engine 
	engine = pyttsx3.init() 
	engine.say(command) 
	engine.runAndWait() 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/textanalysis")
def index():
    fname = request.args.get('fname')
    userText = fname
    results = model.predict([bag_of_words(userText, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["pintents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    reply = random.choice(responses)
    datta =  reply
    return render_template("indexx.html",data=datta)
    return render_template("index.html")

@app.route("/videoanalysis")
def index():
    fname = request.args.get('fname')
    command2mp3 = "ffmpeg -i "+fname+" aurd.mp3"
    command2wav = "ffmpeg -i aurd.mp3 aurd.wav"
    os.system(command2mp3)
    os.system(command2wav)
    r = sr.Recognizer()
    audio = sr.AudioFile('aurd.wav')
    datta= r.recognize_google(audio)
    userText = datta
    results = model.predict([bag_of_words(userText, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["pintents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    reply = random.choice(responses)
    datta =  reply
    return render_template("indexx.html",data=datta)

@app.route("/imageanalysis")
def index():
    fname = request.args.get('fname')
    img = Image.open(fname)
    pytesseract.pytesseract.tesseract_cmd ='C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    result = pytesseract.image_to_string(img)
    userText = result
    results = model.predict([bag_of_words(userText, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["pintents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    reply = random.choice(responses)
    datta =  reply
    return render_template("indexx.html",data=datta)

@app.route("/audioanalysis")
def index():
    fname = request.args.get('fname')
    r = sr.Recognizer()
    audio = sr.AudioFile(fname)
    datta= r.recognize_google(audio)
    userText = datta
    results = model.predict([bag_of_words(userText, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["pintents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    reply = random.choice(responses)
    datta =  reply
    return render_template("indexx.html",data=datta)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/query")
def query():
    return render_template("query.html")

@app.route("/market")
def market():
    df = pandas.read_csv('mark1.csv')
    df2 = pandas.read_csv('mark2.csv')
    ct = df['r1'].count()
    ct2 = df2['r1'].count()
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    for i in range(ct):
        data1.append(df['r1'][i])
        data2.append(df['r2'][i])
        data3.append(df['r3'][i])
    for i in range(ct2):
        data4.append(df2['r1'][i])
        data5.append(df2['r2'][i])
        data6.append(df2['r3'][i])
    return render_template("market.html", data=[data1, data2, data3, data4, data5, data6])

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/feedsubmit")
def feedsubmit():
	df = pandas.read_csv('feedback.csv')
	ct = df['name'].count()
	namecsv = []
	emailcsv = []
	contactcsv = []
	desccsv = []
	statuscsv = []
	for i in range(ct):
		namecsv.append(df['name'][i])
		emailcsv.append(df['email'][i])
		contactcsv.append(df['contact'][i])
		desccsv.append(df['desc'][i])
		statuscsv.append(df['status'][i])

	name = request.args.get('name')
	email = request.args.get('email')
	contact = request.args.get('contact')
	desc = request.args.get('desc')
	url = "https://text-sentiment.p.rapidapi.com/analyze"
	headers = {
		'content-type': "application/x-www-form-urlencoded",
		'x-rapidapi-key': "8f063108cfmsh3aa100a3fcfbaacp154179jsnb2004b15c7fc",
		'x-rapidapi-host': "text-sentiment.p.rapidapi.com"
		}
	response = requests.request("POST", url, data=desc, headers=headers)
	print(response.text)
	with open('feedback.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["name", "email", "contact", "desc", "status"])
		for i in range(ct):
			writer.writerow([namecsv[i], emailcsv[i], contactcsv[i], desccsv[i], statuscsv[i]])
		writer.writerow([name, email, contact, desc, response.text])
	return render_template("/feedsubmit.html")

@app.route("/get")


#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    results = model.predict([bag_of_words(userText, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    reply = random.choice(responses)
    return reply
  
@app.route("/audio")
def listen_user_sudio_frommic():
    while(1):	 
        
        # Exception handling to handle 
        # exceptions at the runtime 
        try: 
            
            # use the microphone as source for input. 
            with sr.Microphone() as source2: 
                
                # wait for a second to let the recognizer 
                # adjust the energy threshold based on 
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2) 
                
                #listens for the user's input 
                audio2 = r.listen(source2) 
                
                # Using ggogle to recognize audio 
                MyText = r.recognize_google(audio2) 
                MyText = MyText.lower() 

                return(MyText) 
                # SpeakText(MyText) 
                
        except sr.RequestError as e: 
            return("Could not request results; {0}".format(e)) 
            
        except sr.UnknownValueError: 
            return("unknown error occured")

if __name__ == "__main__":
    app.run()
