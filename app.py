import streamlit as st
from utils import (
    load_faq,
    generate_embeddings,
    create_faiss_index,
    search_faq,
    initialize_langchain_model,
    generate_faq_response
)
import os
from dotenv import load_dotenv

# Définir TOKENIZERS_PARALLELISM pour éviter les avertissements
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API OpenAI depuis les variables d'environnement via secrets
# Note : Ne pas utiliser os.getenv directement pour les secrets sur Streamlit
# Utilisez st.secrets à la place
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None


# Titre principal de l'application
st.title('🤖 Chatbot FAQ Professionnel')

# Champ de saisie pour les questions des utilisateurs
prompt = st.text_input('Posez votre question ❓', key="prompt")

# Bouton de soumission
submit = st.button("Obtenir une Réponse")

# Fonction mise en cache pour charger les FAQ
@st.cache_data
def get_faq_data():
    return load_faq()


if submit:
    if openai_api_key:

        # Charger les FAQ (mis en cache)
        faq_data = get_faq_data()
        st.write("Chargement des FAQ terminé...") # A commenter

        # Générer les embeddings
        embeddings, model, dimension = generate_embeddings(faq_data)

        # Créer l'index FAISS
        index, answers = create_faiss_index(embeddings, dimension, faq_data)
        st.write("Index FAISS créé avec succès...") # A commenter

        # Récupérer les documents pertinents depuis FAISS
        relavant_docs = search_faq(prompt, index, answers, model, k=2)
        st.write("Documents pertinents trouvés :")
        st.write(relavant_docs)

        # Initialiser le modèle LangChain avec OpenAI
        llm = initialize_langchain_model(openai_api_key, model_name="gpt-3.5-turbo")  # ou "gpt-4" si disponible
        st.write("Modèle de langage initialisé avec LangChain et OpenAI...")

        # Générer la réponse en utilisant le modèle de langage avec LangChain
        response = generate_faq_response(relavant_docs, prompt, llm)
        st.success("Réponse à votre question :")
        st.write(response)
    else:
        st.sidebar.error("Ooopssss!!! La clé API OpenAI est manquante dans le fichier .env.....")
