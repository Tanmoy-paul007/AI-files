# **ঘন্টা ১: এআই-এর জন্য পাইথন ফান্ডামেন্টালস (৬০ মিনিট)**  

## **শেখার উদ্দেশ্য**  
# - এআই ডেভেলপমেন্টের জন্য পাইথন এনভায়রনমেন্ট সেট আপ করা  
# - পাইথন সিনট্যাক্সের বেসিক বিষয়গুলো বুঝা  
# - এআই-এর জন্য গুরুত্বপূর্ণ ডাটা টাইপ নিয়ে কাজ করা  

## **আলোচ্য বিষয়**  

### **১. এনভায়রনমেন্ট সেটআপ (১৫ মিনিট)**  
# - **পাইথন ও আনাকোন্ডা ইন্সটল করা**  
#   - পাইথন একটি প্রোগ্রামিং ল্যাঙ্গুয়েজ, আনাকোন্ডা হলো পাইথনের জন্য একটি ডাটা সায়েন্স প্যাকেজ।  
# - **জুপিটার নোটবুক সেটআপ**  
#   - জুপিটার নোটবুক একটি ইন্টারেক্টিভ টুল যেখানে কোড লিখে রান করা যায় এবং ফলাফল দেখতে পারা যায়।  
# - **গুরুত্বপূর্ণ লাইব্রেরি পরিচিতি**  
#   - **NumPy**: সংখ্যা ও অ্যারে নিয়ে কাজ করার জন্য  
#   - **Pandas**: ডাটা অ্যানালাইসিসের জন্য  
#   - **Matplotlib**: ডাটা ভিজুয়ালাইজেশনের জন্য  
#   - **Scikit-learn**: মেশিন লার্নিং মডেল বানানোর জন্য  

### **২. পাইথন বেসিক রিভিউ (৩০ মিনিট)**  

#### **ভেরিয়েবল ও ডাটা টাইপ**  
# ```python
name = "AI Workshop"  # স্ট্রিং (টেক্সট ডাটা)
age = 25              # ইন্টিজার (পূর্ণ সংখ্যা)
height = 5.8          # ফ্লোট (দশমিক সংখ্যা)
is_student = True     # বুলিয়ান (True/False)
# ```

#### **লিস্ট ও ডিকশনারি (ডাটা হ্যান্ডলিংয়ের জন্য জরুরি)**  
# ```python
features = [1.2, 3.4, 2.1, 4.5]  # লিস্ট (একাধিক ডাটা সংরক্ষণ করে)
data_point = {"age": 25, "income": 50000, "education": "Bachelor"}  # ডিকশনারি (কী-ভ্যালু পেয়ার)
# ```

#### **ফাংশন (এআই কোড রিইউজেবল করার জন্য)**  
# ```python
def calculate_accuracy(correct, total):
    return (correct / total) * 100  # একিউরেসি ক্যালকুলেট করে
# ```

#### **লিস্ট কম্প্রিহেনশন (ডাটা প্রসেসিংয়ের জন্য শক্তিশালী)**  
# ```python
squared_features = [x**2 for x in features]  # প্রতিটি সংখ্যার বর্গ করে নতুন লিস্ট বানায়
# ```

### **৩. হ্যান্ডস-অন এক্সারসাইজ (১৫ মিনিট)**  
# **একটি সাধারণ ডাটা প্রসেসিং ফাংশন তৈরি করুন যা সংখ্যার লিস্ট থেকে বেসিক স্ট্যাটিস্টিক্স (গড়, সর্বোচ্চ, সর্বনিম্ন) বের করে।**  

#### **উদাহরণ সমাধান:**  
# ```python
data = [10, 20, 30, 40, 50]

def result(data):
     average = sum(data) / len(data)  
     maximum = max(data)                 
     minimum = min(data)   
     return {"Avg ": average, "Max ": maximum, "Min ": minimum}

result(data)