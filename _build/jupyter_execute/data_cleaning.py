#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# ## Import Dependencies 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# ## Import Data

# In[2]:


df = pd.read_excel('/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Mobile.de/zsf_031021_mobile_data_1000_200000.xlsx')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# ### Clean Column "price"

# In[6]:


#Alle Zeilen entfernen, bei denen kein Preis vorhanden ist
df = df[df['price'] != '-1']


# In[7]:


df.price


# In[8]:


# Zwei Datentypen (int, str) in der Spalte "price"
df.price.apply(type).value_counts()


# In[9]:


df_str = df[df.price.apply(type) == str]
df_int = df[df.price.apply(type) == int]


# In[10]:


# Entfernen von ., € und Brutto
df_str['price'] = df_str['price'].apply(lambda x: float(x.lower().replace('(brutto)', '').replace('.','').replace('€','')))
df_int.price = df_int.price.astype(float)
df_new = pd.concat([df_str, df_int], axis=0)


# In[11]:


df_new


# ### Create Modelname from Carname

# In[12]:


def make_car_model(x):
    xl = x.lower()
    if 'a-klasse' in xl or 'a klasse' in xl or ' A ' in x or 'A1' in x or 'A2' in x or 'A4' in x:
        return 'A-Klasse'
    
    elif 'b-klasse' in xl or 'b klasse' in xl or ' B ' in x or 'B1' in x or 'B2' in x:
        return 'B-Klasse'
    
    elif 'c-klasse' in xl or 'c klasse' in xl or ' c ' in xl or 'c1' in xl or 'c2' in xl or 'c3' in xl:
        return 'C-Klasse'
    
    elif 'e-klasse' in xl or 'e klasse' in xl or ' E ' in x or 'e2' in xl or 'e3' in xl:
        return 'E-Klasse'
    
    elif 's-klasse' in xl or 's klasse' in xl or ' S ' in x:
        return 'S-Klasse'
    
    elif 'g-klasse' in xl or 'g klasse' in xl or ' G ' in x:
        return 'G-Klasse'
    
    elif 'm-klasse' in xl or 'm klasse' in xl or 'ML' in x or 'ml' in xl:
        return 'M-Klasse'
    
    elif 'x-klasse' in xl or 'x klasse' in xl:
        return 'X-Klasse'
    
    elif 'r-klasse' in xl or 'r klasse' in xl or ' r ' in xl:
        return 'R-Klasse'
    
    elif 'v-klasse' in xl or 'v klasse' in xl or ' v ' in xl or 'viano' in xl:
        return 'V-Klasse'
    
    elif 'marco polo' in xl or 'marco' in xl or 'polo' in xl:
        return 'MarcoPolo'
    
    elif 'cla' in xl:
        return 'CLA'
    
    elif 'clc' in xl:
        return 'CLC'
    
    elif 'clk' in xl:
        return 'CLK'
    
    elif 'cl' in xl:
        return 'CL'
    
    elif 'cls' in xl:
        return 'CLS'
    
    elif 'sl' in xl:
        return 'SL'
    
    elif 'slc' in xl:
        return 'SLC'
    
    elif 'slk' in xl:
        return 'SLK'
    
    elif 'slr' in xl:
        return 'SLR'
    
    elif 'sls' in xl:
        return 'SLS'
    
    elif 'gla' in xl:
        return 'GLA'
    
    elif 'glb' in xl:
        return 'GLB'
    
    elif 'glc' in xl:
        return 'GLC'
    
    elif 'glk' in xl:
        return 'GLK'
    
    elif 'gle' in xl:
        return 'GLE'
    
    elif 'gls' in xl:
        return 'GLS'
    
    elif 'gl' in xl:
        return 'GL'
    
    elif 'GT' in x:
        return 'GT'
    
    elif 'vaneo' in xl or 'citan' in xl:
        return 'Vaneo'
    
    elif 'vito' in xl:
        return 'Vito'
    
    elif 'sprinter' in xl:
        return 'Sprinter'
    
    else:
        return 'OTHER'


# In[13]:


df_new.carname.apply(type).value_counts()
df_new[df_new.carname.apply(type) == int]
df_new = df_new.drop(660)


# In[14]:


df_new['Model'] = df_new['carname'].apply(make_car_model)


# In[15]:


df_new


# ### Clean column "milage"

# In[16]:


df_new['milage'] = df_new['milage'].apply(lambda x: float(x.lower().replace('km', '').replace('.','')))


# ### Clean column "hubraum"

# In[17]:


df_str = df_new[df_new.hubraum.apply(type) == str]
df_int = df_new[df_new.hubraum.apply(type) == int]


# In[18]:


df_int['hubraum'] = df_int['hubraum'].replace(-1, np.nan)
df_str['hubraum'] = df_str['hubraum'].str.replace(' cm³', '')
df_str['hubraum'] = df_str['hubraum'].astype(float)
df_new = pd.concat([df_str, df_int], axis=0)


# In[19]:


df_new


# ### Clean column power PS

# In[20]:


df_new.power.apply(type).value_counts()
df_new[df_new.power.apply(type) == int]
df_new['power'] = df_new['power'].replace(-1, np.nan)
df_new['power'] =  df_new['power'].astype(str)


# In[21]:


df_new.info()
df_new.power.apply(type).value_counts()


# In[22]:


df_new['power_ps'] = df_new['power'].apply(lambda x: x.lower().replace('(', '').split()[2] if x != "nan" else None)


# In[23]:


df_new


# ### Clean column power KW

# In[24]:


df_new['power_kw'] = df_new['power'].apply(lambda x: x.lower().replace('(', '').split()[0] if x != "nan" else None)


# ### Clean column fuel_type

# In[25]:


df_new.fuel_type.apply(type).value_counts()


# In[26]:


df_new['fuel_type'] = df_new['fuel_type'].astype(str)


# In[27]:


def fueltype(x):
    if x == '-1':
        
        return np.nan
    else:
        return x.split()[0].replace(',', '')


# In[28]:


df_new['fuel_type'] = df_new['fuel_type'].apply(fueltype)


# ### Clean column transmission

# In[29]:


df_new['transmission'].value_counts().plot(kind='bar')


# In[30]:


df_new['transmission'] = df_new['transmission'].replace(-1, np.nan)


# In[31]:


df_new.info()


# ### Calculate Age of the Car from frist_registration

# In[32]:


def car_age(x):
    year = float(x.split('/')[1])
    month = float(x.split('/')[0])
    age = 2021.83 - year - (month/12.0)
    return age


# In[33]:


df_new['age'] = df_new['first_registration'].apply(car_age)


# ### Clean column num_seats

# In[34]:


df_new['num_seats'] = df_new['num_seats'].apply(lambda x: str(x) if x != -1 else np.nan)


# In[35]:


df_new['num_seats'].value_counts().plot(kind='bar')


# In[36]:


df_new.loc[df['num_seats'] == 8]


# ### Clean column num_doors

# In[37]:


df_new['num_doors'] = df_new['num_doors'].apply(lambda x: x if x != -1 else np.nan)


# In[38]:


df_new


# ### Clean column emission_class

# In[39]:


df_new['emission_class'] = df_new['emission_class'].apply(lambda x: x if x != -1 else np.nan)


# In[40]:


df_new['emission_class'] = df_new['emission_class'].replace('Euro6d-TEMP', 'Euro6d')


# In[41]:


df_new['emission_class'].value_counts().plot(kind='bar')


# ### Clean column car_type

# In[42]:


def simplify_cartype(x):
    xl = x.lower()
    xl = xl.replace(',','')
    xl = xl.replace('tageszulassung', '').replace('vorführfahrzeug', '')
    if 'suv' in xl:
        return 'suv'
    elif 'van' in xl or 'minibus' in xl:
        return 'van'
    elif 'cabrio' in xl or 'roadster' in xl:
        return 'cabrio'
    elif 'sportwagen' in xl or 'coup' in xl:
        return 'sport'
    elif 'limousine' in xl:
        return 'limousine'
    elif 'kombi' in xl:
        return 'kombi'
    elif 'kleinwagen' in xl:
        return 'kleinwagen'
    elif 'andere' in xl:
        return 'andere'
    else:
        return xl


# In[43]:


df_new['car_type'] = df_new['car_type'].apply(simplify_cartype)


# ### Clean column num_owners

# In[44]:


df_new['num_owners'] = df_new['num_owners'].apply(lambda x: str(x) if x != -1 else np.nan)


# ### Clean column damage

# In[45]:


df_new['damage'] = df_new['damage'].astype(str)


# In[46]:


def damage(x):
    xl = x.lower()
    if 'repariert' in xl:
        return 'repariert'
    elif 'unfallfrei' in xl:
        return 'unfallfrei'
    else:
        return np.nan


# In[47]:


df_new.fuel_type.apply(type).value_counts()


# In[48]:


df_new['schaden'] = df_new['damage'].apply(damage)


# In[49]:


df_new


# ### Remove columns that are no longer needed

# In[50]:


df_cleaned = df_new.drop(['construction_year', 'power', 'first_registration', 'damage'], axis=1)


# In[51]:


df_cleaned.info()


# ### Change dtypes of the columns

# In[52]:


df_cleaned['fuel_type'] = df_cleaned['fuel_type'].astype("category")
df_cleaned['transmission'] = df_cleaned['transmission'].astype("category")
df_cleaned['num_seats'] = df_cleaned['num_seats'].astype("category")
df_cleaned['num_doors'] = df_cleaned['num_doors'].astype("category")
df_cleaned['emission_class'] = df_cleaned['emission_class'].astype("category")
df_cleaned['car_type'] = df_cleaned['car_type'].astype("category")
df_cleaned['num_owners'] = df_cleaned['num_owners'].astype("category")
df_cleaned['Model'] = df_cleaned['Model'].astype("category")
df_cleaned['power_ps'] = df_cleaned['power_ps'].astype(float)
df_cleaned['power_kw'] = df_cleaned['power_kw'].astype(float)
df_cleaned['schaden'] = df_cleaned['schaden'].astype("category")


# In[53]:


df_cleaned


# In[ ]:





# In[ ]:




