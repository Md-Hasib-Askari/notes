# 📘 Feature Engineering → Domain-Specific Features

Domain-specific feature engineering means **tailoring features to the type of data** (text, images, time series, etc.). Different data types need specialized transformations before ML models can use them effectively.

---

## **1. Text Data Features**

Text is **unstructured** and must be converted into numerical representation.

* **Bag of Words (BoW)**

  * Represents text as word counts.
  * Example: “I love data” → {I:1, love:1, data:1}.
  * Simple but ignores word order/context.

* **TF-IDF (Term Frequency–Inverse Document Frequency)**

  * Weighs words by importance: frequent in a document but rare in the corpus → higher score.
  * Better than BoW for reducing noise (e.g., common words like “the”).

* **Word Embeddings**

  * Dense vector representations capturing meaning.
  * Examples: Word2Vec, GloVe, FastText.
  * Similar words → close in vector space.

* **Contextual Embeddings**

  * Advanced embeddings that consider context (e.g., “bank” in “river bank” vs “money bank”).
  * Examples: BERT, GPT embeddings.
  * State-of-the-art for NLP tasks.

📌 Use cases: sentiment analysis, spam detection, document classification.

---

## **2. Image Data Features**

Images are high-dimensional (e.g., 256×256 pixels = 65,536 features), so feature extraction is critical.

* **Raw Pixels**

  * Used in simple ML, but inefficient.

* **Handcrafted Features**

  * **HOG (Histogram of Oriented Gradients)** → captures edge directions.
  * **SIFT (Scale-Invariant Feature Transform)** → detects keypoints invariant to scale/rotation.

* **Deep Learning Features**

  * CNNs (Convolutional Neural Networks) learn hierarchical features:

    * Early layers → edges, textures.
    * Later layers → shapes, objects.
  * Pretrained models (ResNet, VGG, EfficientNet) used as **feature extractors**.

📌 Use cases: facial recognition, medical imaging, object detection.

---

## **3. Time Series Data Features**

Time series data has **temporal dependencies** (order matters).

* **Statistical Features**

  * Mean, variance, rolling averages, autocorrelation.

* **Fourier Transform (FFT)**

  * Converts time domain → frequency domain.
  * Helps detect cycles (e.g., seasonality in sales).

* **Seasonality & Trend Decomposition**

  * Break down series into **trend, seasonality, residuals**.
  * Example: Sales = long-term growth (trend) + holiday spikes (seasonal).

* **Lag & Window Features**

  * Previous values as predictors (lags).
  * Rolling windows for smoothing.

📌 Use cases: stock prediction, demand forecasting, sensor anomaly detection.

---

## ✅ Key Takeaways

1. **Text** → BoW, TF-IDF, embeddings (Word2Vec, BERT).
2. **Images** → HOG, SIFT, CNN features, transfer learning.
3. **Time Series** → statistical, Fourier, seasonal decomposition, lags.
4. Domain-specific features often **outperform generic ones**, especially in specialized tasks.

---