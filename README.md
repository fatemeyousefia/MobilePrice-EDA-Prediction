# Mobile Price Classification | EDA & Prediction

Exploratory data analysis and classification models that predict a mobile phone's price range (low, medium, high, very high) from its technical specifications.

Result: a tuned linear SVM predicts the price range with 97.0% accuracy (macro-F1 0.97) on a held-out test set. RAM is by far the strongest driver of price.


> Completed as part of the **Data Science & ML Course (IMT)** by Mohamadreza Momeni.

## Business Question

Product and pricing teams need to know which specifications actually separate a low-cost phone from a premium one. This project answers that in two steps:

1. **Analysis:** which specs are related to the price range, and which are not?
2. **Prediction:** can the price range (0 = low, 1 = medium, 2 = high, 3 = very high) be predicted from the specs alone?

## Dataset

- Source: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) on Kaggle.
- `train.csv`: 2,000 phones, 20 specifications + `price_range` (4 balanced classes, 500 phones each).
- `test.csv`: 1,000 phones without a price range, used only at the end to generate predictions.
- No missing values and no duplicates, but some implausible values (for example 180 phones with a screen width of 0 cm).

The dataset is not included in this repository. Download it from Kaggle and create a data/ folder containing train.csv and test.csv.

| Feature group | Columns |
|---|---|
| Performance | `ram`, `clock_speed`, `n_cores`, `int_memory` |
| Battery | `battery_power`, `talk_time` |
| Screen & size | `px_height`, `px_width`, `sc_h`, `sc_w`, `m_dep`, `mobile_wt` |
| Cameras | `fc` (front), `pc` (primary) |
| Connectivity & features | `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi` |

## Approach

1. **Data cleaning:** missing values, duplicates, unique values and suspicious values (zero screen width, very small pixel heights). Rows were kept, because removing them would have cost up to 36% of the data.
2. **EDA:** target distribution, binary features, battery, camera, memory, RAM, resolution, talk time, correlation analysis.
3. **Feature check:** three feature sets compared with 5-fold stratified cross-validation on the training set, instead of dropping columns by intuition.
4. **Modeling:** Decision Tree, Random Forest and SVM, tuned with `GridSearchCV`. The SVM scaler sits inside a `Pipeline` to avoid data leakage.
5. **Evaluation:** one stratified 80/20 split, the final model chosen by cross-validated accuracy (not by test score), then evaluated once on the test set with accuracy, macro-F1, confusion matrices and a classification report.

## Key Findings

- **RAM is the strongest driver of price range.** The four price classes are clearly separated by RAM.
- Battery power and pixel resolution show a weaker positive relationship with price.
- Weight, talk time, screen size, processor speed and most on/off features (Bluetooth, dual SIM, Wi-Fi, ...) show almost no relationship with price.
- A low correlation was not a good reason to drop a column: cross-validation showed that keeping `px_height` improved the models, even though its linear correlation with price is only 0.149.
- The classes are close to linearly separable, which is why the linear SVM beats the more flexible models.

## Model Results

| Model | CV Accuracy | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| Decision Tree | 0.844 | 0.852 | 0.853 |
| Random Forest | 0.884 | 0.905 | 0.905 |
| **SVM (linear)** | **0.966** | **0.970** | **0.970** |


## Project Structure

```
.
├── notebook/
│   └── MobilePrice-EDA-Prediction.ipynb
├── data/                  # train.csv and test.csv (download from Kaggle)
├── requirements.txt
└── README.md
```

## Skills Demonstrated

Data cleaning and validation · exploratory data analysis · data visualization (Plotly, Seaborn, Matplotlib) · feature engineering · cross-validated model comparison (scikit-learn) · communicating findings

## How to Run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook notebook/MobilePrice-EDA-Prediction.ipynb
```

Make sure `train.csv` and `test.csv` are in the `data/` folder before running the notebook.

## Limitations

- The values look noisy or partly synthetic, so the results should not be read as a real market study.
- The price range is a class label from 0 to 3, not a real price.

## Tools & Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn


## Author

**Fatemeh Yousefi Amiri**

[GitHub](https://github.com/fatemeyousefia) · [LinkedIn](https://www.linkedin.com/in/fatemeyousefi)
