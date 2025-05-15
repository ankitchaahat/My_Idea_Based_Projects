# 🧠 Personality Detector from WhatsApp Chats (MBTI Classifier)

Predict someone's MBTI personality type (e.g. INTP, ENFJ) based on their exported WhatsApp chat — using real NLP and machine learning!

---

## 📌 Why This Project?

- 🎯 **Real-world application**: Works on your friends' chats (with consent!)
- 🔐 **Privacy-centric**: No data stored or shared.
- 🤖 **Fun + Insightful**: Get to know your personality in a data-driven way!

---

## 💡 How It Works

1. Export a WhatsApp chat `.txt` file.
2. Clean the text (automated).
3. Use TF-IDF vectorization + 4 ML classifiers to predict:
   - I/E → Introvert vs Extrovert  
   - N/S → Intuitive vs Sensing  
   - T/F → Thinking vs Feeling  
   - J/P → Judging vs Perceiving  
4. Final result → MBTI type like `INTP`, `ENFJ`, etc.
5. Also shows confidence % for each axis.
6. Bonus: Spider chart visualization of personality dimensions.

---

## ⚙️ Tech Stack

- Python 🐍
- scikit-learn
- pandas
- matplotlib
- Jupyter Notebook / Streamlit (optional dashboard)
- MBTI dataset (open-source, 8k+ rows)

---

## 📂 Dataset

Used [Kaggle MBTI Personality Dataset](https://www.kaggle.com/datasnaek/mbti-type)  
Also cleaned posts column with advanced preprocessing:
- Removed stopwords, links, emojis, MBTI-type mentions, short words, etc.
- Lowercased and tokenized

---

## 📈 Sample Output

