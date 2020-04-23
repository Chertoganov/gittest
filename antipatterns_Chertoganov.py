#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
urls = [
    "https://us.burberry.com/womens-new-arrivals-new-in/",
    "https://us.burberry.com/womens-new-arrivals-new-in/?start=2&pageSize=120&productsOffset=&cellsOffset=8&cellsLimit=&__lang=en"
]

# SCRAPING & CREATING A LIST OF LINKS
doc = []
for url in urls:
    r = requests.get(url)
    html_doc = r.text
    soup = BeautifulSoup(html_doc)

    for link in soup.find_all("a"):
        l = link.get("href")
        if "-p80" in l: # <-- THIS WILL NEED TO CHANGE
            doc.append(l)

# DEDUPLICATING THE LIST OF LINKS
doc_uniq = set(doc)
print("Number of unique items:"+str(len(doc_uniq)))

# CREATING A DICTIONARY WITH WORDS : COUNTS AND KEY : VALUE PAIRS
result = {}
for link in doc_uniq:
    words = link.replace("/", "").split("-")
    for word in words:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1
            
words = list(result.keys())
counts = list(result.values())

# TURNING THE DICTIONARY INTO A DATAFRAME, SORTING & SELECTING FOR RELEVANCE
df = pd.DataFrame.from_dict({
    "words": words,
    "counts": counts,
})

df_sorted = df.sort_values("counts", ascending = True)
df_rel = df_sorted[df_sorted['counts']>3]
print(df_rel.head())
print(df_rel.shape) 

# PLOTTING
plt.barh(df_rel['words'], df_rel['counts'], color = "#C19A6B")
plt.title("Most used words in Burberry 'New in' SS2020 Women collection")
plt.xticks(np.arange(0, 18, step=2))
plt.savefig("SS2020_Burberry_word_frequency.jpg")
df_rel['brand']='burberry'

df_burberry = df_rel


# In[2]:


# VERSACE

# CREATING LIST OF RELEVANT URLS
url = "https://www.versace.com/us/en-us/women/new-arrivals/new-in/"
    
# SCRAPING & CREATING A LIST OF LINKS
doc = []
#for url in urls:
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
soup_f = soup.find_all("a")
for t in soup_f:
    a = t.get("href")
    if a.startswith("/us/en-us/women/new-arrivals/new-in/"):
        doc.append(a)


# DEDUPLICATING THE LIST OF LINKS
doc_uniq = set(doc)
print("Number of unique items:"+str(len(doc_uniq)))
#print(doc_uniq)

result = {}
garbage = []
for link in doc_uniq:
    if link.startswith("/us/en-us/women/new-arrivals/new-in/?"):
        continue
    words = link.replace("/us/en-us/women/new-arrivals/new-in/", "") .split("/")
    words = words[0].split("-")
    
    for word in words:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1

words = list(result.keys())
counts = list(result.values())
#print(result)

# TURNING THE DICTIONARY INTO A DATAFRAME, SORTING & SELECTING FOR RELEVANCE
df = pd.DataFrame.from_dict({
    "words": words,
    "counts": counts,
})

df2 = df.set_index("words")
#df2 = df.drop(["a1008"],axis=0)
df_sorted = df2.sort_values("counts", ascending = True)
df_rel = df_sorted[df_sorted['counts']>2]
#print(df_rel.head())
#print(df_rel.shape) 

#PLOTTING
plt.barh(df_rel.index, df_rel['counts'], color = "#FFD700")
plt.title("Most used words in Versace 'New in' SS2020 Women collection")
plt.savefig("SS2020_Versace_word_frequency.jpg")
df_rel['brand']='versace'

df_versace = df_rel


# In[3]:


# CREATING LIST OF RELEVANT URLS
urls = []
#urls = list(urls)
for i in [1,2,3,4]:
    u = str("https://us.dolcegabbana.com/en/women/highlights/new-in/?page=") + str(i)
    urls.append(u)
    
#print(urls)

# SCRAPING & CREATING A LIST OF LINKS
doc = []
for url in urls:
    r = requests.get(url)
    html_doc = r.text
    soup = BeautifulSoup(html_doc)
    soup_f = soup.find_all("a")
    
    for t in soup_f:
        a = t.get("aria-label")
        if a != None and a.startswith("Visit"):
            doc.append(a)
#print(doc)

# DEDUPLICATING THE LIST OF LINKS
doc_uniq = set(doc)
print("Number of unique items:"+str(len(doc_uniq)))

result = {}
for link in doc_uniq:
    words = link.replace("Visit", "").replace(" product page","").split(" ")                                                                                                                                      
    for word in words:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1
del(result[""])
words = list(result.keys())
counts = list(result.values())

# TURNING THE DICTIONARY INTO A DATAFRAME, SORTING & SELECTING FOR RELEVANCE
df = pd.DataFrame.from_dict({
    "words": words,
    "counts": counts,
})

df2 = df.set_index("words")
#df2.drop(["", "WITH"])
df_sorted = df2.sort_values("counts", ascending = True)
df_rel = df_sorted[df_sorted['counts']>4]
#print(df_rel.head())
#print(df_rel.shape) 


# PLOTTING
plt.barh(df_rel.index, df_rel['counts'], color = "#E0115F")
plt.title("Most used words in D&G 'New in' SS2020 Women collection")
plt.savefig("SS2020_D&G_word_frequency.jpg", pad_inches=0.1)
df_rel['brand']='d&g'

df_dg = df_rel


# In[4]:


df_brands = pd.concat([df_versace.reset_index(), df_burberry.reset_index(), df_dg.reset_index()])


# In[5]:


df_brands = df_brands.drop(columns=['index'])


# In[6]:


df_brands['words'] = df_brands['words'].str.upper()


# In[7]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[8]:


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df_brands['words'])
print(integer_encoded)


# In[9]:


# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# In[10]:


def brand_target(brand):
    if 'versace' in brand:
        val = 0
    elif 'burberry' in brand:
        val = 1
    elif 'd&g' in brand:
        val = 2
    else:
        raise ValueError(f'Invalid brand: {brand}')
    return val


# In[11]:


def apply_brand_loop(df):
    brand_list = []
    for i in range(len(df)):
        brand = df.iloc[i]['brand']
        target = brand_target(brand)
        brand_list.append(target)
    return brand_list


# In[12]:


brands_transformed = apply_brand_loop(df_brands.copy())
y = brands_transformed

X = onehot_encoded


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


#models = []


# In[17]:


#from sklearn.dummy import DummyClassifier


#models.append(("Baseline", DummyClassifier(random_state=391488407)))


# In[18]:


#from sklearn.svm import SVC
#models.append(("Support vector (C=1.30)", SVC(C=1.30, gamma="scale")))
#models.append(("Support vector (C=0.45)", SVC(C=0.45, gamma="scale")))
#models.append(("Support vector (C=4.80)", SVC(C=4.80, gamma="scale")))
#models.append(("Support vector (C=1.00)", SVC(C=1.00, gamma="scale")))
#models.append(("Support vector (C=0.25)", SVC(C=0.25, gamma="scale")))
#models.append(("Support vector (C=4.00)", SVC(C=4.00, gamma="scale")))


# In[19]:


#from sklearn.ensemble import RandomForestClassifier


#models.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=1283220422)))
#from sklearn.linear_model import LogisticRegression


#models.append(("Logistic Regression (C=1.00)",
 #              LogisticRegression(C=1.00, solver="liblinear", penalty="l1")))

#models.append(("Logistic Regression (C=0.25)",
 #              LogisticRegression(C=0.25, solver="liblinear", penalty="l1")))

#models.append(("Logistic Regression (C=4.00)",
  #             LogisticRegression(C=4.00, solver="liblinear", penalty="l1")))


# In[20]:


#from sklearn.neighbors import KNeighborsClassifier


#models.append(("1-nn euclidean",
    #           KNeighborsClassifier(n_neighbors=1)))

#models.append(("1-nn cosine",
    #           KNeighborsClassifier(n_neighbors=1, metric="cosine")))

#models.append(("5-nn cosine",
     #          KNeighborsClassifier(n_neighbors=5, metric="cosine")))


# In[21]:


#models = dict(models)


# ## Changed

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

base_clfs = [GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[1, 2, 3, 4, 5]}), 
             GridSearchCV(LogisticRegression(), {'C':np.linspace(1, 11, 10)}),
             GridSearchCV(SVC(), {'C':np.linspace(1, 0.25, 4)})] 
for clf in base_clfs:
    clf.fit(X_train, y_train)
    print('Best parameters:{}', clf.best_params_)


# In[24]:


from sklearn.metrics import accuracy_score

names = ['kNN', 'Logistic Regression', 'SVC']
base_clfs = [KNeighborsClassifier(n_neighbors=1),
             LogisticRegression(C=1), 
             SVC(C=0.5)]
base_scores = []
for clf, name in zip(base_clfs, names):
    clf.fit(X_train, y_train)
    base_scores = np.append(base_scores, accuracy_score(clf.predict(X_test), y_test))
scores = base_scores


# In[25]:


print('KNN: ',scores[0] , 'Logistic Regression: ', scores[1],'SVC: ', scores[2])


# In[ ]:


#scores = pd.Series({name: clf.score(X_test, y_test) for name, clf in models.items()}, name="Accuracy")


# In[61]:


#scores

