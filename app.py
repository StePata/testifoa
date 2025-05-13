import streamlit as st


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def create_pairplot(data, features, hue=None):
    """Crea un pairplot"""
    fig = sns.pairplot(data[features], hue=hue, diag_kind='kde', height=2)
    return fig

def plot_hist_qq(df):
    """Genera histplot e QQ-plot"""
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df[column], kde=True, ax=axs[0])
        axs[0].set_title(f'Distplot of {column}')

        stats.probplot(df[column], dist="norm", plot=axs[1])
        axs[1].set_title(f'QQ-Plot of {column}')

        st.pyplot(fig)
        plt.clf()

def plot_heatmap(df):
    """Genera una heatmap delle variabili numeriche"""
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    plt.title("Matrice di Correlazione")
    st.pyplot(fig)
    plt.clf()

def main():
    st.title(" EDA + Classificazione")

    uploaded_file = st.file_uploader("Scegli un file CSV o XLSX", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
            st.success("File CSV caricato!")
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=None)
            st.success("File XLSX caricato!")
        else:
            st.error("Formato non supportato!")
            return

        column_names = st.text_input("Inserisci i nomi delle colonne (separati da virgola)", 
                                    value="sepal_length,sepal_width,petal_length,petal_width,species")
        col_names_list = [name.strip() for name in column_names.split(',')]

        if len(col_names_list) == df.shape[1]:
            df.columns = col_names_list
            st.success("Colonne rinominate correttamente!")
        else:
            st.error(f"Numero colonne errato! (Servono {df.shape[1]} colonne)")

        df = df[1:].reset_index(drop=True)  # Salta la prima riga se necessario
        st.dataframe(df)

        hue = df.columns[-1]  # Ultima colonna come target

        features = st.multiselect('Seleziona le colonne per Pairplot:', df.columns.tolist())
        if features:
            fig = create_pairplot(df, features, hue=hue)
            st.pyplot(fig)

        st.subheader("Histplot e QQ-Plot delle Variabili")
        plot_hist_qq(df)

        st.subheader("Heatmap della Correlazione")
        plot_heatmap(df)

        st.subheader("Classificazione")

        # Seleziona X (feature) e y (target)
        X = df.drop(columns=[hue])
        y = df[hue]

        # Label Encoding per target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Divisione train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,stratify= y_encoded)

        # Lista dei classificatori
        classifiers = [
            DecisionTreeClassifier(max_depth=4),
            RandomForestClassifier(n_estimators=200, random_state=667),
            GradientBoostingClassifier(),
            GradientBoostingClassifier(n_estimators=50),
            LogisticRegression(max_iter=1000),
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=2),
            SVC(probability=True, kernel='rbf'),
            XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        ]

        classifier_names = [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Gradient Boosting (n_estimators=50)",
            "Logistic Regression",
            "Gaussian Naive Bayes",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "XGBoost"
        ]

        selected_classifiers = st.multiselect("Scegli i classificatori da allenare:", classifier_names, default=classifier_names)

        for clf_name, clf in zip(classifier_names, classifiers):
            if clf_name in selected_classifiers:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.text(classification_report(y_test, preds))
                st.write(f"**{clf_name}** - Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()