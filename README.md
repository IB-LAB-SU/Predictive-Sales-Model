# 📊 Dual-Objective Retail Revenue & Profit Predictive Engine

An end-to-end machine learning pipeline and interactive dashboard designed to solve a critical retail optimization problem: contrasting the structural drivers of top-line **Sales Amount** versus bottom-line **Net Profit**.

Live Interactive Dashboard: *[Insert your Streamlit Cloud Link Here]*

---

## 🎯 Project Overview

In retail analytics, optimizing purely for transaction volume can cause businesses to overlook aggressive promotional strategies that cannibalize margins. This project deploys a **sequential dual-modeling architecture** using `scikit-learn` to isolate input features—such as unit prices, shipping costs, regional metrics, and discount percentages—and map their true mathematical influence side-by-side against revenue and profit targets.

### Key Insights Discovered
* **Premium Scale:** High `unit_price` items and the `electronics` category serve as the primary structural revenue engines.
* **Promotion Elasticity:** The model revealed that while `discount_pct` has a significantly lower impact weight on profit than sales volume, it remained positive—proving that promotional incentives actively scaled total transaction volume without dipping below profitable baseline margins.
* **Volume Paradox:** Higher bulk order `quantity` negatively correlated with overall transactional margins, indicating a high-volume saturation of lower-value commodity goods (e.g., groceries).

---

## 🛠️ Technical Architecture & Pipeline

### 1. Data Engineering & Sanitization
* **Text Standardization:** Applied case-insensitive normalization across nominal values (`gender`, `region`, `product_category`, etc.) to eliminate string-fragmentation errors.
* **Imputation Strategy:** Handled missing tracking blocks natively (e.g., creating explicit indicator flags for missing shipping metrics and filling gaps via statistical medians) to prevent row-dropping data loss.
* **Categorical Encoding:** Converted high-cardinality values into discrete mathematical feature spaces via binary dummy switches to prevent false mathematical hierarchies.

### 2. Model Training & Scaling
* **Feature Normalization:** Utilized `StandardScaler` to perform $Z$-score normalization across input arrays, completely eliminating magnitude bias between heavily skewed features (e.g., small individual quantities vs. large unit costs).
* **Validation Split:** Enforced a strict data partition sequence to separate validation data from training data, entirely eliminating structural data leakage during the preprocessing phase.
* **Algorithm:** Multi-variable Ordinary Least Squares (OLS) Linear Regression, yielding a **58.17% $R^2$ score** for the Sales Model and a **49.79% $R^2$ score** for the Profit Model.

---

## 💻 Tech Stack

* **Core Engine:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visual Frontend:** Streamlit, Plotly Express

---

## 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
