# 📘 Beginner → Data Types in Machine Learning

Machine Learning models rely on data as input. The **type of data** determines how it should be stored, processed, and transformed before training. Broadly, ML data falls into **three categories**: structured, semi-structured, and unstructured.

---

## **1. Structured Data**

* **Definition**: Data that is organized into rows and columns (tabular format) with a well-defined schema. Each row represents an observation (record), and each column represents a feature (attribute).
* **Examples**:

  * Banking transactions (amount, date, account ID)
  * Sales records (product, price, quantity, customer ID)
  * Healthcare records (age, blood pressure, diagnosis)
* **Characteristics**:

  * Easy to store in relational databases (SQL).
  * Clean and consistent format.
  * Numeric and categorical values dominate.
* **Challenges**:

  * Missing values or inconsistencies.
  * Feature correlations may cause redundancy.
* **ML Usage**:

  * Works well with classical algorithms like **Linear/Logistic Regression, Decision Trees, Random Forests, XGBoost, KNN**.
  * Example: Predicting house prices based on size, location, and number of rooms.

---

## **2. Semi-Structured Data**

* **Definition**: Data that does not follow a strict tabular structure but still carries some organizational properties (tags, keys, or metadata).
* **Examples**:

  * **JSON/XML**: `{ "user": "Alice", "purchase": "Book", "price": 15 }`
  * **Logs**: `2025-09-04 08:10:23 | ERROR | Connection timeout`
  * Email metadata: sender, recipient, timestamp (but body is unstructured).
* **Characteristics**:

  * Flexible schema (fields may vary across records).
  * Common in APIs, web apps, and NoSQL databases.
  * Can combine structured & unstructured parts (e.g., metadata + free text).
* **Challenges**:

  * Requires **parsing and normalization** before ML use.
  * May contain nested or inconsistent fields.
* **ML Usage**:

  * Converted to structured format (e.g., flatten JSON into tables).
  * Useful in **recommendation systems, fraud detection, log anomaly detection**.
  * Example: Predicting website errors from server log patterns.

---

## **3. Unstructured Data**

* **Definition**: Data without a predefined schema or structure. It’s the most abundant type of data in the real world.
* **Examples**:

  * **Text**: Articles, social media posts, chat conversations.
  * **Images**: X-rays, satellite images, selfies.
  * **Audio**: Music, speech recordings, call center logs.
  * **Video**: Security camera feeds, movies, sports footage.
* **Characteristics**:

  * Rich and complex, often high-dimensional.
  * Cannot be directly fed into ML models without transformation.
* **Challenges**:

  * Requires **feature extraction** or representation learning.
  * Large storage requirements and computationally expensive.
* **ML Usage**:

  * Text → NLP methods (Bag of Words, TF-IDF, Word2Vec, BERT).
  * Images → Computer Vision (CNNs, transfer learning).
  * Audio → Spectrograms, MFCCs, speech recognition models.
  * Video → Sequence models (RNN, Transformers) or frame-based analysis.
  * Example: Sentiment analysis on tweets, facial recognition, speech-to-text.

---

## **Comparison Table**

| Data Type       | Storage Format     | Examples                    | ML Approach                             |
| --------------- | ------------------ | --------------------------- | --------------------------------------- |
| Structured      | Tables, SQL        | Sales data, health records  | Classical ML (Regression, Trees)        |
| Semi-Structured | JSON, XML, Logs    | Web logs, emails, API data  | Parsed + Tabular ML / anomaly detection |
| Unstructured    | Text, Image, Audio | Tweets, X-rays, voice notes | Deep Learning (NLP, CV, Speech models)  |

---

✅ **Key Takeaways**:

* Structured data is the **easiest** to use for ML.
* Semi-structured data is **flexible** but requires transformation.
* Unstructured data is the **richest** but requires **advanced ML (deep learning)** to extract features.
