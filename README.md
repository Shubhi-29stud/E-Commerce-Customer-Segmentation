# E-Commerce Customer Segmentation using Unsupervised Learning

An end-to-end data science case study applying **Unsupervised Learning** and **Dimensionality Reduction** techniques to segment an online retailer's customer base using **RFM (Recency, Frequency, Monetary)** analysis.

---

## 📂 Project Structure
- `ecommerce_rfm_dataset.csv`: The customer transaction and RFM feature dataset.
- `customer_segmentation.py` / Jupyter Notebook: Complete implementation code covering data exploration, feature scaling, evaluation metrics, K-Means, DBSCAN, PCA, and t-SNE.

---

## ⚙️ Methodology & Implementation Steps

1. **Feature Engineering & Exploration:**
   * Analyzed the distribution of `Recency`, `Frequency`, and `Monetary` values using histograms with KDE curves.
   * Engineered an additional behavioral feature: **Average Order Value (AOV)** ($\text{Monetary} / \text{Frequency}$).

2. **Data Preprocessing:**
   * **Log Transformation (`np.log1p`):** Applied to handle the heavy right-skewness typical in retail monetary and recency distributions.
   * **Feature Scaling (`StandardScaler`):** Standardized features to zero mean and unit variance to ensure equal distance weighting during clustering.

3. **Optimal Cluster Selection ($K=3$):**
   * **Elbow Method:** Inspected the Within-Cluster Sum of Squares (Inertia) to identify the natural point of inflection.
   * **Silhouette Score Analysis:** Evaluated cluster separation across $K = 2 \text{ to } 6$. The silhouette score peaks at **$K = 3$ (~0.2705)**, mathematically confirming it as the optimal partition size for this dataset.

4. **Clustering Models:**
   * **K-Means ($K=3$):** Partition-based clustering dividing the customer base into three distinct macro-segments.
   * **DBSCAN:** Density-based spatial clustering configured to isolate data noise/outliers (anomalies labeled as `-1`).

5. **Dimensionality Reduction & Visualization:**
   * Reduced high-dimensional features to 2D space using **PCA** (Principal Component Analysis) and visualized cluster distributions cleanly using **t-SNE** (t-Distributed Stochastic Neighbor Embedding).

---

## 📊 Cluster Profiles & Actionable Marketing Personas ($K=3$)

| Cluster ID | Recency (Days) | Frequency (Orders) | Monetary Spend ($) | AOV ($) | Customer Count | Assigned Persona / Strategy |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | 47.56 | 3.26 | $264.19 | $105.46 | 449 | **Standard / Casual Shoppers:** Nurture with cross-selling and second-purchase incentives. |
| **Cluster 1** | 69.24 | 5.87 | $1,282.98 | $230.08 | 634 | **At-Risk / Hibernating VIPs:** High-value historical spenders requiring targeted win-back campaigns. |
| **Cluster 2** | 8.61 | 5.22 | $992.52 | $202.02 | 417 | **Active Champions:** Highly engaged, recent buyers to be rewarded with VIP loyalty perks. |

---

## 🛠️ Tech Stack
* **Python**
* **Pandas & NumPy** (Data Manipulation & Numerical Operations)
* **Scikit-Learn** (K-Means, DBSCAN, PCA, t-SNE, StandardScaler, Silhouette Score)
* **Matplotlib & Seaborn** (Data Visualization & Plotting)

---

## 🚀 How to Run the Code
1. Clone the repository and ensure your dataset (`ecommerce_rfm_dataset.csv`) is in the same directory.
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn