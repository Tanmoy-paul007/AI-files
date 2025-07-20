# **ঘন্টা ৪: এআই-এর জন্য ডাটা ভিজ্যুয়ালাইজেশন (৬০ মিনিট)**  

## **শেখার উদ্দেশ্য**  
# - ডাটার প্যাটার্ন বুঝতে ভিজ্যুয়ালাইজেশন তৈরি করা  
# - মডেল পারফরম্যান্স মেট্রিক্স প্লট করা  
# - এআই রেজাল্ট কার্যকরভাবে উপস্থাপন করা  

## **আলোচ্য বিষয়**  

### **১. Matplotlib বেসিকস (৩০ মিনিট)**  
# **Matplotlib** হলো পাইথনের সবচেয়ে জনপ্রিয় ডাটা ভিজ্যুয়ালাইজেশন লাইব্রেরি।  

#### **সিম্পল প্লট তৈরি**  
# ```python
import matplotlib.pyplot as plt

# ডাটা প্রস্তুত
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# লাইন প্লট
plt.plot(x, y, color='green', marker='o')
plt.title("সরল রেখার উদাহরণ")
plt.xlabel("X-অক্ষ")
plt.ylabel("Y-অক্ষ")
plt.grid(True)
plt.show()
# ```

#### **বার চার্ট ও হিস্টোগ্রাম**  
# ```python
# বার চার্ট
labels = ['Python', 'Java', 'C++']
values = [70, 50, 30]
plt.bar(labels, values, color='blue')
plt.title("প্রোগ্রামিং ভাষার জনপ্রিয়তা")

# হিস্টোগ্রাম
data = [1, 1, 2, 3, 3, 3, 4, 4, 5]
plt.hist(data, bins=5, color='red')
plt.title("ডাটা বিন্যাস")
# ```

# ---

### **২. Seaborn দিয়ে অ্যাডভান্সড ভিজ্যুয়ালাইজেশন (২০ মিনিট)**  
# **Seaborn** Matplotlib-এর উপর ভিত্তি করে তৈরি আরও সুন্দর ও সহজ ভিজ্যুয়ালাইজেশনের লাইব্রেরি।  

#### **ডাটাসেট লোড করে ভিজ্যুয়ালাইজেশন**  
# ```python
import seaborn as sns
tips = sns.load_dataset("tips")  # বিল্ট-ইন ডাটাসেট

# স্ক্যাটার প্লট
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="time")
plt.title("টিপ vs বিল")

# বক্স প্লট (আউটলিয়ার ডিটেকশনের জন্য)
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("প্রতিদিনের বিলের বণ্টন")
# ```

#### **হিটম্যাপ (করিলেশন ম্যাট্রিক্স)**  
# ```python
# শুধুমাত্র সংখ্যাসূচক কলামগুলোর মধ্যে সম্পর্ক বের করা
corr = tips.select_dtypes(include='number').corr()

# হিটম্যাপ আঁকা
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("কলামগুলোর মধ্যে সম্পর্ক")
plt.show()
# ```

# ---

### **৩. হ্যান্ডস-অন এক্সারসাইজ (১০ মিনিট)**  
# **একটি ডাটাসেটের জন্য কম্প্রিহেনসিভ ভিজ্যুয়ালাইজেশন ড্যাশবোর্ড তৈরি করুন।**  

#### **ধাপসমূহ:**  
# 1. **ডাটা লোড করুন** (`pandas` দিয়ে)  
# 2. **বিভিন্ন প্লট তৈরি করুন** (`Matplotlib` ও `Seaborn` ব্যবহার করে)  
# 3. **সাবপ্লট ব্যবহার করে ড্যাশবোর্ড বানান**  

#### **উদাহরণ:**  
# ```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ডেটাসেট লোড করা
tips = sns.load_dataset('tips')

# 2. ফিগার ও সাবপ্লট তৈরি
plt.figure(figsize=(15, 10))  # ফিগারের আকার

# 1ম সাবপ্লট: মোট বিলের হিস্টোগ্রাম
plt.subplot(2, 2, 1)
sns.histplot(tips['total_bill'], bins=20, kde=True, color='skyblue')
plt.title("মোট বিলের হিস্টোগ্রাম")

# 2য় সাবপ্লট: টিপ বনাম মোট বিল (স্ক্যাটার প্লট)
plt.subplot(2, 2, 2)
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.title("টিপ বনাম মোট বিল")

# 3য় সাবপ্লট: বক্সপ্লট (স্মোকার বনাম টিপ)
plt.subplot(2, 2, 3)
sns.boxplot(data=tips, x='smoker', y='tip', palette='Set2')
plt.title("স্মোকার অনুযায়ী টিপ বিতরণ")

# 4র্থ সাবপ্লট: হিটম্যাপ (সংখ্যাসূচক কলামগুলোর মধ্যে সম্পর্ক)
plt.subplot(2, 2, 4)
corr = tips.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("সংখ্যাসূচক কলামের সম্পর্ক")

# সব প্লট ঠিকভাবে দেখানোর জন্য লেআউট ঠিক করা
plt.tight_layout()
plt.suptitle("Tips Dataset Visualization Dashboard", fontsize=16, y=1.02)
plt.show()

# ```
# 🧠 এই ড্যাশবোর্ড কী দেখায়?
# সাবপ্লট	ব্যাখ্যা
# 1️⃣	মোট বিলের ডিস্ট্রিবিউশন (histogram)।
# 2️⃣	মোট বিল আর টিপের মধ্যে সম্পর্ক (scatter plot)।
# 3️⃣	স্মোকার এবং নন-স্মোকারদের টিপ দেওয়ার বক্সপ্লট।
# 4️⃣	সংখ্যা ভিত্তিক কলামগুলোর মধ্যে পারস্পরিক সম্পর্ক (correlation heatmap)।
# ---

# ## **কেন এটি গুরুত্বপূর্ণ?**  
# ✅ **ডাটা প্যাটার্ন বোঝা** (যেমন: আউটলিয়ার, ডিস্ট্রিবিউশন)  
# ✅ **মডেল পারফরম্যান্স বিশ্লেষণ** (যেমন: কনফিউশন ম্যাট্রিক্স, ROC কার্ভ)  
# ✅ **স্টেকহোল্ডারদের কাছে রেজাল্ট সহজে বোঝানো**  

# **পরবর্তী সেশনে আমরা মেশিন লার্নিং মডেল বিল্ডিং শিখব!** 🚀