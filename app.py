import streamlit as st
import pandas as pd
import xgboost as xgb  # Importer XGBoost

# Titre de l'application
st.title("Prédiction d'Attrition des Clients")

model_path = "xgb_model.bin"
try:
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_path)
    st.success("Modèle XGBoost chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Étape 1 : Téléchargement du fichier CSV
st.header("Téléchargez le fichier CSV")

required_columns = {
    "Customer_Age": "Âge du client en années.",
    "Dependent_count": "Nombre de personnes dépendantes du client.",
    "Months_on_book": "Durée (en mois) depuis que le client est actif.",
    "Total_Relationship_Count": "Nombre total de relations avec la banque.",
    "Months_Inactive_12_mon": "Nombre de mois inactifs au cours des 12 derniers mois.",
    "Contacts_Count_12_mon": "Nombre de contacts avec le client au cours des 12 derniers mois.",
    "Credit_Limit": "Limite de crédit du client.",
    "Total_Revolving_Bal": "Solde total des crédits renouvelables.",
    "Avg_Open_To_Buy": "Moyenne des crédits ouverts disponibles.",
    "Total_Amt_Chng_Q4_Q1": "Variation du montant des transactions entre T4 et T1.",
    "Total_Trans_Amt": "Montant total des transactions au cours des 12 derniers mois.",
    "Total_Trans_Ct": "Nombre total de transactions au cours des 12 derniers mois.",
    "Total_Ct_Chng_Q4_Q1": "Variation du nombre de transactions entre T4 et T1.",
    "Avg_Utilization_Ratio": "Ratio moyen d'utilisation du crédit.",
    "Complain": "1 si le client a déposé une plainte, sinon 0.",
    "Satisfaction Score": "Score de satisfaction du client (1 à 5).",
    "Point Earned": "Points fidélité gagnés par le client.",
    "Gender_M": "1 si le client est un homme, sinon 0.",
    "Education_Level_Doctorate": "1 si niveau d'éducation = Doctorat, sinon 0.",
    "Education_Level_Graduate": "1 si niveau d'éducation = Graduate, sinon 0.",
    "Education_Level_High School": "1 si niveau d'éducation = Lycée, sinon 0.",
    "Education_Level_Post-Graduate": "1 si niveau d'éducation = Post-Graduate, sinon 0.",
    "Education_Level_Uneducated": "1 si niveau d'éducation = Non éduqué, sinon 0.",
    "Education_Level_Unknown": "1 si niveau d'éducation est inconnu, sinon 0.",
    "Marital_Status_Married": "1 si le statut marital est marié, sinon 0.",
    "Marital_Status_Single": "1 si le statut marital est célibataire, sinon 0.",
    "Marital_Status_Unknown": "1 si le statut marital est inconnu, sinon 0.",
    "Income_Category_$40K - $60K": "1 si revenu entre 40K et 60K, sinon 0.",
    "Income_Category_$60K - $80K": "1 si revenu entre 60K et 80K, sinon 0.",
    "Income_Category_$80K - $120K": "1 si revenu entre 80K et 120K, sinon 0.",
    "Income_Category_Less than $40K": "1 si revenu inférieur à 40K, sinon 0.",
    "Income_Category_Unknown": "1 si revenu est inconnu, sinon 0.",
    "Card_Category_Gold": "1 si la catégorie de carte est Gold, sinon 0.",
    "Card_Category_Platinum": "1 si la catégorie de carte est Platinum, sinon 0.",
    "Card_Category_Silver": "1 si la catégorie de carte est Silver, sinon 0."
}

# Affichage de la liste des colonnes obligatoires
st.write("Assurez-vous que votre fichier CSV contient les colonnes suivantes avec leurs descriptions :")
column_descriptions = "".join([f"- {col}: {desc}\n" for col, desc in required_columns.items()])
st.text_area("", column_descriptions, height=300)

    
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire les données du CSV
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données téléchargées :")
    st.write(data.head())
    st.write("Shape des données d'entrée :", data.shape)

    # Étape 2 : Prétraitement des données si nécessaire
    st.header("Prétraitement des données")
    
    # Supposons que seules certaines colonnes sont nécessaires pour le modèle
    required_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Complain', 'Satisfaction Score', 'Point Earned', 'Gender_M', 'Education_Level_Doctorate', 'Education_Level_Graduate', 'Education_Level_High School', 'Education_Level_Post-Graduate', 'Education_Level_Uneducated', 'Education_Level_Unknown', 'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Unknown', 'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K', 'Income_Category_Less than $40K', 'Income_Category_Unknown', 'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'] 
    if all(col in data.columns for col in required_columns):
        st.success("Toutes les colonnes nécessaires sont présentes.")
        data_scaled = data[required_columns]

        # Normalisation (si votre modèle a besoin de données normalisées)
        #scaler = StandardScaler()
        #data_scaled = scaler.fit_transform(data_scaled)
    else:
        st.error(f"Colonnes manquantes. Assurez-vous que le CSV contient : {required_columns}")
        st.stop()
    

    st.header("Prédictions")
    probabilities = loaded_model.predict_proba(data_scaled)
    predictions = loaded_model.predict(data_scaled)

    data["Proba_Classe_0"] = probabilities[:, 0]  # Probabilité de la classe 0
    data["Proba_Classe_1"] = probabilities[:, 1] 
    data["AttritionPrediction"] = predictions

    # Affichage des résultats
    st.write("Résultats des prédictions :")
    st.write(data[["CLIENTNUM", "AttritionPrediction", "Proba_Classe_0", "Proba_Classe_1"]])

    # Étape 5 : Visualisation (optionnelle)
    st.header("Visualisation des résultats")
    st.bar_chart(data["AttritionPrediction"].value_counts())