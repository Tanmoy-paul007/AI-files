import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x,y, color="green", marker="O")
plt.title("সরল রেখার উদাহরণ")
plt.xlabel("X-অক্ষ")
plt.ylabel("Y-অক্ষ")
plt.grid(True)
plt.show()


labels = ['Python', 'Java', 'C++']
values = [70, 50, 30]
plt.bar(labels, values, color="green")
plt.title("প্রোগ্রামিং ভাষার জনপ্রিয়তা")

data = [1, 1, 2, 3, 3, 3, 4, 4, 5]
plt.hist(data, bins=5, color='red')
plt.title("ডাটা বিন্যাস")

tips = sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", data=tips, hue="time")
plt.title("টিপ vs বিল")

sns.boxenplot(x="day", y="total_bill", data=tips)
plt.title("প্রতিদিনের বিলের বণ্টন")

corr = tips.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("কলামগুলোর মধ্যে সম্পর্ক")
plt.show()


