# **ঘন্টা ৫: মেশিন লার্নিং পরিচিতি (৬০ মিনিট)**

## **শেখার উদ্দেশ্য**
# - মেশিন লার্নিংয়ের মৌলিক ধারণা বোঝা
# - সহজ কিছু ML অ্যালগরিদম বাস্তবায়ন করা
# - মডেলের পারফরম্যান্স মূল্যায়ন করা

## **আলোচ্য বিষয়**

### **১. মেশিন লার্নিং ফান্ডামেন্টালস (২০ মিনিট)**
#### **সুপারভাইজড vs আনসুপারভাইজড লার্নিং**
# - **সুপারভাইজড লার্নিং**: টার্গেট/আউটপুট জানা থাকে (যেমন: ইনকাম প্রেডিকশন)
# - **আনসুপারভাইজড লার্নিং**: শুধু ইনপুট ডাটা থাকে, কোন টার্গেট নেই (যেমন: গ্রুপিং)

#### **ট্রেনিং ও টেস্টিং ডাটা**
# - **ট্রেনিং ডাটা**: মডেল ট্রেন করতে ব্যবহার (সাধারণত ৭০-৮০%)
# - **টেস্ট ডাটা**: মডেল টেস্ট করতে ব্যবহার (সাধারণত ২০-৩০%)

#### **সাধারণ অ্যালগরিদম**
# - **রিগ্রেশন**: সংখ্যাগত মান প্রেডিক্ট করা (যেমন: বাসার দাম)
# - **ক্লাসিফিকেশন**: ক্যাটাগরি প্রেডিক্ট করা (যেমন: ইমেইল স্প্যাম/নট স্প্যাম)

# ---

### **২. হ্যান্ডস-অন ML ইমপ্লিমেন্টেশন (৩৫ মিনিট)**

#### **রিগ্রেশন উদাহরণ (Linear Regression)**
# ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ডাটা প্রস্তুত
X = df[['age', 'education_years']]  # ফিচার
y = df['income']  # টার্গেট

# ডাটা বিভক্ত
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# মডেল ট্রেন
model = LinearRegression()
model.fit(X_train, y_train)

# প্রেডিকশন
predictions = model.predict(X_test)

# মূল্যায়ন
mse = mean_squared_error(y_test, predictions)
print(f"গড় বর্গ ত্রুটি: {mse}")
# ```

#### **ক্লাসিফিকেশন উদাহরণ (Random Forest)**
# ```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# মডেল ট্রেন
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# একিউরেসি চেক
accuracy = accuracy_score(y_test, classifier.predict(X_test))
print(f"একিউরেসি: {accuracy:.2f}")
# ```

# ---

### **৩. মডেল ইভ্যালুয়েশন (৫ মিনিট)**
#### **ক্রস-ভ্যালিডেশন**
# - ডাটাকে কয়েক ভাগে ভাগ করে বারবার ট্রেন-টেস্ট করা

#### **পারফরম্যান্স মেট্রিক্স**
# - **রিগ্রেশন**: Mean Squared Error (MSE), R² Score
# - **ক্লাসিফিকেশন**: Accuracy, Precision, Recall

#### **ওভারফিটিং এড়ানো**
# - বেশি কমপ্লেক্স মডেল ট্রেনিং ডাটায় ভালো কিন্তু টেস্ট ডাটায় খারাপ
# - সমাধান: ডাটা বাড়ানো, ফিচার কমানো, Regularization ব্যবহার

# ---

## **কেন এটি গুরুত্বপূর্ণ?**
# ✅ **রিয়েল-ওয়ার্ল্ড প্রব্লেম সলভ** (যেমন: স্টক প্রেডিকশন, রোগ শনাক্তকরণ)  
# ✅ **ডাটা থেকে প্যাটার্ন শেখা**  
# ✅ **অটোমেটেড ডিসিশন মেকিং**  

# **পরবর্তী সেশনে আমরা ডিপ লার্নিং ও নিউরাল নেটওয়ার্ক শিখব!** 🚀