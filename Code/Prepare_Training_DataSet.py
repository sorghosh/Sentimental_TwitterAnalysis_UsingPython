import nltk
import os
import pandas as pd
import pickle


stopwords       = set(nltk.corpus.stopwords.words())
otherstopwords  = set([",", "." ,'``',"?","!",":","''"])
stemmer         = nltk.stem.wordnet.WordNetLemmatizer() 
document_temp   = []
all_words_temp  = []
train_dataset   = {}
file_counter    = 0
def preparedataset(filename,cat_type):
    f = open(filename,"rb")
    line = f.readline()
    while line:
        try:
            line = f.readline().strip("\n")
            document_temp.append((line,cat_type))
            words = nltk.word_tokenize(line)
            words = [w for w in words if w not in stopwords]
            words = [w for w in words if w not in otherstopwords]
            words = [w.lower() for w in words if len(w) >2]
            words = [stemmer.lemmatize(w) for w in words] 
            for w in words:
                all_words_temp.append(w)
        except Exception:
            pass
    f.close()    
    return document_temp,all_words_temp


def get_features(doc):
    words = nltk.word_tokenize(doc)
    for w in word_features:
        train_dataset[w] = (w in words)
    return train_dataset

def save_file(output_dir, dataset,category,file_counter):
    df = pd.DataFrame(dataset.values()).T
    df.columns = dataset.keys()
    df["labels"] = category
    filename = output_dir + "Training_Dataset"+".txt"
    
    if os.path.isfile(filename) and file_counter == 0:
        with open(filename,"a") as f:
            f.truncate()
            df.to_csv(f,header = True,index = False)
        f.close()
        file_counter = file_counter + 1
    elif os.path.isfile(filename) and file_counter > 0 :
        with open(filename,"a") as f:
            df.to_csv(f,header = False , index = False)
        f.close()
        file_counter = file_counter + 1
    else:
        with open(filename,"a") as f:
            df.to_csv(f,header = True , index = False)
        f.close()
        file_counter = file_counter + 1
    
    return file_counter
    

Training_Dataset = [(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\BaseLineData\negative.txt",0),(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\BaseLineData\positive.txt",1)]
output_dir       = r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output/"

for t in Training_Dataset:
    document,all_words = preparedataset(t[0],t[1])
    
word_features = list(nltk.FreqDist(all_words))[:8000]
save_document = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\word_features.pickle","wb")
pickle.dump(word_features,save_document)
save_document.close()

for d,cat in document:
    try:
        f =  get_features(d)
        file_counter = save_file(output_dir,f,cat,file_counter)
    except Exception:
        pass
    