
# coding: utf-8

# # Data Mining Project: Explorative Analysis of mobile.de car prices

# Considering we thought about a project where we had to scrap data in order to create a model, one of our first choices was the creation of a model for second hand car prices. 
# 
# Second hand cars webpages contain a lot of different information for each one of the cars, so I would not be difficult to query that sites and extract all the meaninful information that we want. After a first analysis of several webpages was conducted, we stucked with mobile.de.

# In[8]:

import sys
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib2
from threading import Thread
from queue import Queue
import math
import requests
import time 
from datetime import timedelta
import pickle
import json


# In[9]:

def extractValue(tag):
    if(len(tag) <= 0):
        return np.nan
    return tag[0].stripped_strings.next()


# In[10]:

def extractInformation(resultID):
    success = False    
    while not success:
        try:
            url = urllib2.urlopen("http://suchen.mobile.de/fahrzeuge/details.html?id=" + str(resultID))
            url = url.read()
            success = True
        except urllib2.HTTPError as e:
            # car not available any more
            return None
        except Exception as e:
            # retry
            success = False
            
    soup = BeautifulSoup(url, "lxml")
    
    # ID does not exist (any longer)
    if len(soup.find_all("div", class_="cBox-body cBox-body--notification-error")) > 0:
        return None
    
    scripts = soup.find_all("script")
    for script in scripts:
        if script.text.startswith("var partnerUrl"):
            text = script.text
            
            text = text[text.find(", make: \"")+9:len(text)]
            make = text[0:text.find("\"")]
            text = text[text.find(", model: '")+10:len(text)]
            model = text[0:text.find("'")]
    
    mileage = extractValue(soup.select("#rbt-mileage-v"))
    power = extractValue(soup.select("#rbt-power-v"))
    fuel = extractValue(soup.select("#rbt-fuel-v"))
    transmission = extractValue(soup.select("#rbt-transmission-v"))
    firstRegistration = extractValue(soup.select("#rbt-firstRegistration-v"))
    damageCondition = extractValue(soup.select("#rbt-damageCondition-v"))
    numSeats = extractValue(soup.select("#rbt-numSeats-v"))
    doorCount = extractValue(soup.select("#rbt-doorCount-v"))
    climatisation = extractValue(soup.select("#rbt-climatisation-v"))
    airbags = extractValue(soup.select("#rbt-airbag-v"))
    color = extractValue(soup.select("#rbt-color-v"))
    interior = extractValue(soup.select("#rbt-interior-v"))
    parkAssist = extractValue(soup.select("#rbt-parkAssists-v"))
    
    price = extractValue(soup.select("span.h3.rbt-prime-price"))
    
    bulletPoints = []
    for bulletPoint in soup.find_all("div", class_="bullet-list"):
        tag = bulletPoint.select("p")
        if len(tag) > 0:
            bulletPoints.append(extractValue(tag))
    
    return [resultID, 
            make,
            model,
            mileage, 
            power, 
            fuel, 
            transmission, 
            firstRegistration, 
            damageCondition, 
            numSeats, 
            doorCount, 
            climatisation, 
            airbags, 
            color, 
            interior, 
            parkAssist,
            price, 
            bulletPoints]


# In[11]:

def buildURL(baseURL):
    url = "?isSearchRequest=true&vc=Car&dam=0&con=USED&ambitCountry=DE"
    # explicitly specify categories to prevent trailers etc. from showing up
    url += "&categories=Cabrio"
    url += "&categories=OffRoad"
    url += "&categories=SmallCar"
    url += "&categories=EstateCar"
    url += "&categories=Limousine"
    url += "&categories=SportsCar"
    url += "&categories=Van"   
    
    return baseURL + url

def buildURLParameters(mileageFrom = -1,
                    mileageTo = -1,
                    firstRegistrationFrom = -1,
                    firstRegistrationTo = -1,
                    priceFrom = -1,
                    priceTo = -1,
                    powerFrom = -1,
                    powerTo = -1):
        
    parameters = ""
    if(mileageFrom >= 0):
        parameters += "&minMileage=" + repr(mileageFrom)
    if(mileageTo >= 0):
        parameters += "&maxMileage=" + repr(mileageTo)
    if(firstRegistrationFrom >= 0):
        parameters += "&minFirstRegistrationDate=" + repr(firstRegistrationFrom) + "-01-01"
    if(firstRegistrationTo >= 0):
        parameters += "&maxFirstRegistrationDate=" + repr(firstRegistrationTo) + "-12-31"
    if(priceFrom >= 0):
        parameters += "&minPrice=" + repr(priceFrom)
    if(priceTo >= 0):
        parameters += "&maxPrice=" + repr(priceTo)
        
    return parameters


# In[12]:

def getResultCount(url):
    nResults = -1
    while nResults == -1:
        try:
            response = requests.get(url)
            json = response.json()
            nResults = json['numResultsTotal']
        except ConnectionError as e:
            print "\nConnection Error for query " + url + ", retrying."
            nResults = -1
    return nResults


# In[ ]:

relevantIDs = set()

def scrapeResultList(baseURL, nResults):
    pageCount = int(math.ceil(nResults / 20.0))
    for i in range(pageCount):
        url = urllib.urlopen(baseURL + "&pageNumber=" + repr(i + 1))
        url = url.read()
        soup = BeautifulSoup(url, "lxml")
        div_results = soup.find_all("div", class_="cBox-body cBox-body--resultitem")

        for div_result in div_results:
            relevantIDs.add(div_result.a["data-ad-id"])

class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
    def run(self):
        while True:
            try:
                baseURL, nResults = self.queue.get()
                scrapeResultList(baseURL, nResults)
                sys.stdout.write("\r#IDs: " + repr(len(relevantIDs)) +" / #Queue: " + repr(self.queue.qsize()))
            except Exception as e:
                print(e)
            finally:
                self.queue.task_done()

queue = Queue(10)
for x in range(2):
    worker = Worker(queue)
    worker.daemon = True
    worker.start()

base_url_search = buildURL("http://suchen.mobile.de/fahrzeuge/auto")
base_url_json = buildURL("http://suchen.mobile.de/fahrzeuge/count.json")
    
for mileage in range(1, 1501): # mileage between 0 and 1500000, interval 1000
    parametersMileage = buildURLParameters(mileageFrom=(mileage-1)*1000, mileageTo=mileage*1000)
    nResultsMileage = getResultCount(base_url_json + parametersMileage)
    
    if len(relevantIDs) > 10000:
        break;
    
    if nResultsMileage <= 1000:
        queue.put((base_url_search + parametersMileage, nResultsMileage))
    else:
        for firstRegistration in range(1900, 2017): # first registration between 1900 and 2016, interval 1
            parametersRegistration = buildURLParameters(firstRegistrationFrom=firstRegistration, firstRegistrationTo=firstRegistration)
            nResultsRegistration = getResultCount(base_url_json + parametersMileage + parametersRegistration)

            if len(relevantIDs) > 10000:
                break;
                
            if nResultsRegistration <= 1000:
                queue.put((base_url_search + parametersMileage + parametersRegistration, nResultsRegistration))
            else:
                for price in range(1, 1001): # price between 0 and 100000, interval 100
                    #print "price " + repr((price-1)*100) + " - " + repr(price*100)
                    parametersPrice = buildURLParameters(priceFrom=(price-1)*100, priceTo=price*100)
                    nResultsPrice = getResultCount(base_url_json + parametersMileage + parametersRegistration + parametersPrice)
                    
                    if len(relevantIDs) > 10000:
                        break;
                        
                    if nResultsPrice <= 1000:
                        queue.put((base_url_search + parametersMileage + parametersRegistration + parametersPrice, nResultsPrice))
                    else:
                        for priceFine in range((price-1)*100, price*100, 10):
                            parametersPriceFine = buildURLParameters(priceFrom=priceFine, priceTo=priceFine+10)
                            nResultsPriceFine = getResultCount(base_url_json + parametersMileage + parametersRegistration + parametersPriceFine)
                    
                            if len(relevantIDs) > 10000:
                                break;
                                
                            queue.put((base_url_search + parametersMileage + parametersRegistration + parametersPriceFine, nResultsPriceFine))
                    
                            if nResultsPriceFine > 1000:
                                print "\nDROPPING " + repr(nResultsPrice - 1000) + " ELEMENTS"
        
queue.join()


# In[ ]:

print repr(len(relevantIDs))


# In[ ]:

relevantIDsList = list(relevantIDs) # just for output
relevantIDCount = len(relevantIDsList)
testData = []

class ResultScraper(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
    def run(self):
        while True:
            try:
                resultID = self.queue.get()
                result = extractInformation(resultID)
                if result is not None:
                    testData.append(result)
            except Exception as e:
                print(e)
            finally:
                self.queue.task_done()
                
taskQueue = Queue(500)
for x in range(2):
    worker = ResultScraper(taskQueue)
    worker.daemon = True
    worker.start()

resultScrapingStartTime = time.time()
    
for i in range(len(relevantIDsList)):
    sys.stdout.write("\rQueueing " + relevantIDsList[i] + 
                     " (" + repr((i+1)) + 
                     " / " + repr(relevantIDCount) + 
                     ", " + repr(((i + 1.0) / relevantIDCount) * 100) + "%)")
    taskQueue.put(relevantIDsList[i])
    
taskQueue.join()
resultScrapingEndTime = time.time()


# In[ ]:

dataFile = open('testData.pckl', 'wb')
pickle.dump(testData, dataFile)


# In[ ]:

print testData[1000]


# In[ ]:

testDataTMP = []
for i in range(len(testData)):
    testDataTMP.append(testData[i][:18])


# In[ ]:

print testDataTMP[0][17][0]


# In[ ]:

testDF = pd.DataFrame(data=testDataTMP, columns=["CarID", 
                                              "Brand",
                                              "Model",
                                              "Mileage", 
                                              "Power", 
                                              "Fuel", 
                                              "Transmission", 
                                              "Registration",
                                              "Damage",
                                              "Seats",
                                              "Doors",
                                              "Climatisation",
                                              "Airbags",
                                              "Color",
                                              "Interior",
                                              "ParkAssist",
                                              "Price"])
testDF.head()


# In[ ]:

testDF.describe()


# In[ ]:

dfFinal = pd.DataFrame(columns=["CarID", "Mileage", "Power","Brand", "Model", "Fuel", "Transmission", "Registration", "Price"])


# In[ ]:

dfFinal['Mileage'] = testDF['Mileage'].str.split('km').str.get(0).astype(float)
dfFinal['Price'] = testDF['Price'].str.split().str.get(0).replace('.', '')
dfFinal['Price'] = dfFinal['Price'].str.replace('.', '').astype(float)
dfFinal['Power'] = testDF['Power'].str.split().str.get(0).astype(float)
dfFinal['Fuel'] = testDF['Fuel']
dfFinal['Transmission'] = testDF['Transmission']
dfFinal['Registration'] = testDF['Registration']
dfFinal['CarID'] = testDF['CarID']
dfFinal['Brand'] = testDF['Brand']
dfFinal['Model'] = testDF['Model']


# In[ ]:

dfFinal.info(verbose=True)


# ## Data cleaning and transformation
# 
# For the purposes of the study this data is not completely accurate and it should be cleaned and treated for the algorithms to be more accurate. For instance, some of the variables have to be converted to numeric, categories have to be properly aggregated and NaNs should be treated.

# In[ ]:

dfFinal.head()


# In[ ]:

dfFinal['Year'] = dfFinal['Registration'].str.extract('(\d\d\d\d)', expand=True)


# In[ ]:

dfFinal.head()


# In[ ]:

dfFinal.corr()


# We can see Power is the column with the highest correlation with Price. As a preliminary analysis let's see the scatterplot of these two variables:

# In[ ]:

dfFinal.plot(kind='scatter', x='Power', y='Price')


# ### Boxplots
# 
# Since most of the data is qualitative, boxplots showing the distribution of some of the most important categories can be shown to be if there are strong differences between them. (For visualization purposes the outliers were taken away)

# In[ ]:

import seaborn as sb


# In[ ]:

dfFinal['Decade'] = dfFinal['Year'].str.extract('(\d\d\d)', expand=True) + "0"


# In[ ]:

plot = sb.boxplot(x=dfFinal['Decade'], y='Price', data=dfFinal, showfliers=False) 
plot.set_title('Distribution of Prices by decade')


# Here we can see the prices decrease in average over time, but the most important feature is that the variability of prices decreases the youngest the cars are.

# In[ ]:

dfFinal['FuelStd'] = dfFinal['Fuel'].str.split(',').str.get(0)


# In[ ]:

dfFinal.FuelStd.unique()


# In[ ]:

plot = sb.boxplot(orient='h', y=dfFinal['FuelStd'], x='Price', data=dfFinal, showfliers=False)
plot.set_title('Distribution of Prices by Fuel')


# After aggregating the fuel types into more general categories we can see that, for example, diesel cars are more expensive in average than gasoline cars, and both are cheaper than hybrid cars.

# In[ ]:

plot = sb.boxplot(x=dfFinal['Transmission'], y='Price', data=dfFinal, showfliers=False) 
plot.set_title('Distribution of Prices by Transmission')


# In[ ]:

plot = sb.boxplot(orient='h', y=dfFinal['Brand'], x='Price', data=dfFinal, showfliers=False)
plot.set_title('Distribution of Prices by Brand')


# In[ ]:

X = dfFinal[['Mileage','Power']].copy()
X['Ones'] = np.ones(len(dfFinal))
y = dfFinal.Price


# In[1212]:

y = y.values
X = X.values


# In[1213]:

# Function definition is here
def adagrad( X, y, iterations='default' ):
    b = np.zeros(3)
    n=len(y)
    if iterations == 'default':
        size = 10*len(y)
    else: 
        size = iterations
    G = 0
    for j in range(size):
        i = random.randint(0,n-1)
        grad = (y[i]-b.dot(X[i]))*-X[i]
        G = G + grad**2
        b = b - (grad/(G**0.5))
        print grad
    #printing the r2 etc
    rss =  sum((y[i]-b.dot(X[i]))**2 for i in range(n))
    tss = sum((y[i]-y.mean())**2 for i in range(n))
    r_squared = 1.-rss/tss
    mse = rss/n
    print mse, r_squared, b
    return b;


# In[1214]:

adagrad(X, y,2)


# In[1215]:

from sklearn.linear_model import LinearRegression


# In[1216]:

Xnew = dfFinal[['Power']].copy()
y = dfFinal.Price
y=y.values
Xnew['Ones1'] = np.ones(len(dfFinal))
Xnew['Ones2'] = np.ones(len(dfFinal))
Xnew=Xnew.values
prediction=adagrad(Xnew, y)
prediction=Xnew*prediction
prediction=np.sum(prediction,axis=1)


# In[ ]:

Xnew = dfFinal[['Power']].copy()
Xnew['Price_Prediction'] = prediction
Xnew['Price'] = dfFinal.Price
Xnew.head()


# In[ ]:

#--------------------------Boris-Experiment------------------------


# In[1217]:

Xnew = dfFinal[['Power']].copy()
y = dfFinal.Price
y=y.values
Xnew['Ones1'] = np.ones(len(dfFinal))
Xnew['Ones2'] = np.ones(len(dfFinal))
Xnew=Xnew.values
#reg = LinearRegression()
#reg.fit(Xnew,y)
#reg.score(Xnew,y)
#prediction=reg.predict(Xnew)


# In[1218]:

i = 0;
for j in range(len(Xnew)):
    if(math.isnan(Xnew[j][0])):
        i=i+1
print i


# In[1219]:

dfCat = dfFinal.copy()
dfCat.head()


# In[1220]:

dfCat = dfCat.drop('CarID', 1)
dfCat = dfCat.drop('Registration', 1)
#dfCat = dfCat.drop('Year', 1)
dfCat = dfCat.drop('Fuel', 1)
dfCat = dfCat.drop('Decade', 1)


# In[1221]:

yearnummer=dfCat['Year']
pd.to_numeric(yearnummer)
#dfCat = dfCat.drop('Year', 1)


# ### NaN operations
# 
# Now that the data is transformed and gathered, we have to deal with NaN values. there are two cases:
# 
# -Categorical data, where the NaN are transformed into a String ("missing") so they can be treated as a new category in each column.
# 
# -Cuantitative data, where the values will be averaged using the data that we already have.

# In[1222]:

dfClean = dfCat.copy()
dfClean['Damage'] = dfClean['Damage'].fillna('missing')
dfClean['ParkAssist'] = dfClean['ParkAssist'].fillna('missing')
dfClean.ix[:,4:12] = dfClean.ix[:,4:12].fillna('missing')
dfClean['FuelStd'] = dfClean['FuelStd'].fillna('missing')
dfClean.info()


# In[1223]:

#dfDrop = dfCat.dropna(axis=0)

#dfCat.Transmission[pd.isnull(dfCat.Transmission)]  = 'NaN'

#from sklearn import preprocessing
#import numpy as np
#le = preprocessing.LabelEncoder()

#le.fit(dfCat.Transmission)

#list(le.classes_)

#dfCat.Transmission = dfCat.Transmission.apply(le.transform)


# In[1224]:

from sklearn.decomposition import *
def repair_mad(df, n_comp = 4, n_iter = 5, n_remove = None, n_rep = 1):
    num_feat = df.shape[1]
    size = df.shape[0]
    mads = [0.]*num_feat
    cnt = [0]*num_feat
    np.random.seed(191)
    if not n_remove:
        n_rep = 1
    for p in range(n_rep):
        df_prep = df.copy().astype(float)
        if n_remove:
            removed = []
            for i in range(n_remove):
                i = np.random.randint(0,size)
                j = np.random.randint(0,num_feat)
                val = df.iat[i,j]
                df_prep.iat[i,j] = np.nan
                removed.append([i,j,val])
        df_train = df_prep.fillna(df_prep.mean())
        # run PCA and reconstruct data set
        for i in range(n_iter):
            pca = PCA(n_components = n_comp).fit(df_train)
            df_pred = pca.inverse_transform(pca.transform(df_train))
            df_pred = pd.DataFrame(df_pred,columns=df.columns,index=df.index)
            df_train = df_prep.combine_first(df_pred)
        if n_remove:
            for pos in removed:
                diff = pos[2] - df_train.iat[pos[0],pos[1]]
                if not np.isnan(diff):
                    cnt[pos[1]] += 1
                    mads[pos[1]] += 1./cnt[pos[1]]*(abs(diff)-mads[pos[1]])
    if n_remove:
        print pd.DataFrame([mads],index=['MAD'],columns=df_train.columns)
    return df_train


# In[1225]:

from sklearn.tree import *
from sklearn import cross_validation
from sklearn.tree import *
from sklearn.ensemble import *


# In[1226]:

processedData=pd.get_dummies(dfClean)
#processedData=pd.get_dummies(dfCat)
processedData['Year']=yearnummer
repaired=repair_mad(processedData,n_iter=20,n_comp = 10,n_rep = 5)

Xnew = repaired.copy()
Xnew = Xnew.drop('Price', 1)
y = repaired.Price
y=y.values

Xnew['Ones1'] = np.ones(len(repaired))
Xnew['Ones2'] = np.ones(len(repaired))
Xnew=Xnew.values


# In[1227]:

repaired.head()


# In[1228]:

repaired.corr()


# In[1229]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xnew, y, random_state=20)

reg=DecisionTreeRegressor(max_depth=8)
reg.fit(X_train,y_train)
reg.score(X_test,y_test)
#prediction=reg.predict(Xnew)


# In[1230]:

reg = RandomForestRegressor(n_estimators=25, oob_score=True)
reg.fit(Xnew,y)
reg.score(Xnew,y),reg.oob_score_


# In[1231]:

reg=LinearRegression()
reg.fit(Xnew,y)
reg.score(Xnew,y)


# In[ ]:

oobScores=[]
for i in range(1,100):
    reg = RandomForestRegressor(n_estimators=i, oob_score=True)
    reg.fit(Xnew,y)
    arrayToAppend=[]
    arrayToAppend.append(reg.oob_score_)
    arrayToAppend.append(i)
    oobScores.append(reg.oob_score_)
    print i
pd.DataFrame([oobScores],index=['original']).transpose().plot(figsize=(16,4))


# In[1]:




# In[ ]:



