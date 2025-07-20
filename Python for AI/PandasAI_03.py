# **ঘণ্টা ৩: Pandas দিয়ে ডাটা ম্যানিপুলেশন (৬০ মিনিট)**  

## **শেখার উদ্দেশ্য**  
# - ডাটাসেট লোড করে এক্সপ্লোর করা  
# - ডাটা ক্লিনিং ও প্রিপ্রসেসিং করা  
# - মেশিন লার্নিংয়ের জন্য ডাটা প্রস্তুত করা  

## **আলোচ্য বিষয়**  

### **১. Pandas ফান্ডামেন্টালস (২৫ মিনিট)**  

#### **ডাটা লোড করা ও এক্সপ্লোরেশন**  
# ```python
import pandas as pd  # Pandas লাইব্রেরি ইম্পোর্ট

# CSV ফাইল থেকে ডাটা লোড
df = pd.read_csv('dataset.csv')

# ডাটার প্রথম কয়েকটি সারি দেখতে
print(df.head())

# ডাটার ইনফো (কলাম, ডাটা টাইপ, নাল ভ্যালু)
print(df.info())

# সংখ্যাগত ডাটার স্ট্যাটিস্টিক্স (গড়, SD, মিন, ম্যাক্স)
print(df.describe())
# ```

#### **ডাটা সিলেকশন ও ফিল্টারিং**  
# ```python
# ৩০ বছরের কম বয়সী কাস্টমারদের ফিল্টার
young_customers = df[df['age'] < 30]

# নির্দিষ্ট ফিচার (কলাম) সিলেক্ট করা
selected_features = df[['age', 'income', 'education']]
# ```

# ---

### **২. এআই-এর জন্য ডাটা ক্লিনিং (২৫ মিনিট)**  

#### **মিসিং ভ্যালু হ্যান্ডলিং**  
# ```python
# মিসিং ভ্যালুকে গড় দিয়ে ফিল করা
df.fillna(df.mean(), inplace=True)

# অথবা মিসিং ভ্যালু থাকলে সেই সারি ডিলিট করা
df.dropna(inplace=True)
# ```

#### **ক্যাটেগরিক্যাল ডাটা এনকোডিং**  
# ```python
# শিক্ষার স্তরকে নিউমেরিক ভ্যালুতে রূপান্তর (One-Hot Encoding)
df['education_encoded'] = pd.get_dummies(df['education'])
# ```

#### **ফিচার স্কেলিং (সব ফিচারকে একই স্কেলে আনা)**  
# ```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])
# ```

# ---

### **৩. হ্যান্ডস-অন এক্সারসাইজ (১০ মিনিট)**  
# **একটি অগোছালো ডাটাসেট ক্লিন করে মেশিন লার্নিংয়ের জন্য প্রস্তুত করুন।**  

#### **ধাপগুলো:**  
# 1. **ডাটা লোড করুন** (`pd.read_csv`)  
# 2. **মিসিং ভ্যালু চেক করুন** (`df.isnull().sum()`)  
# 3. **মিসিং ভ্যালু ফিল করুন** (`fillna` বা `dropna`)  
# 4. **ক্যাটেগরিক্যাল ডাটা এনকোড করুন** (`pd.get_dummies`)  
# 5. **ফিচার স্কেলিং করুন** (`StandardScaler`)  

#### **উদাহরণ:**  
# ```python

# ```
# ধরি 'messy_data.csv' একটি নোংরা ডাটাসেট
df = pd.read_csv('messy_data.csv')

# মিসিং ভ্যালু চেক
MissValues = df.isnull().sum() 
print("Missingvalues is :- " + MissValues)

# সংখ্যাগত কলামের মিসিং ভ্যালু গড় দিয়ে ফিল
df.fillna(df.mean(), inplace=True)

# ক্যাটেগরিক্যাল ডাটা এনকোড
df = pd.get_dummies(df, colums=['education'])

# ফিচার স্কেলিং
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

Cleaned_Data = df.head()
Cleaned_Data

# ---

## **কেন এটি গুরুত্বপূর্ণ?**  
# - **মেশিন লার্নিং মডেল** ভালোভাবে কাজ করার জন্য **পরিষ্কার ও স্কেলড ডাটা** প্রয়োজন।  
# - Pandas দিয়ে আমরা সহজেই **ডাটা ফিল্টার, ক্লিন ও ট্রান্সফর্ম** করতে পারি।  
# - `sklearn` এর `StandardScaler` সব ফিচারকে একই স্কেলে নিয়ে আসে, যাতে কোনো একটি ফিচার প্রভাবশালী না হয়।  

# পরবর্তী সেশনে আমরা **ডাটা ভিজুয়ালাইজেশন (Matplotlib/Seaborn)** শিখব! 