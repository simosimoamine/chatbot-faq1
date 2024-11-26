import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage



def load_faq():
    """
    Charge les questions fréquentes (FAQ) depuis un fichier JSON.

    Returns:
        list: Une liste de dictionnaires contenant les questions et réponses.
    """
    with open('faq.json', 'r', encoding='utf-8') as file:
        faq_data = json.load(file)
    return faq_data


#Fonction pour extraire les questions stockées dans le fichier json
# et les convertir en vecteurs numériques (embeddings)
def generate_embeddings(faq_data, model_name="all-MiniLM-L6-v2"):
    """
    Génère des embeddings pour les questions de la FAQ en utilisant SentenceTransformer.
    """
    # Charger le modèle d'embedding
    model = SentenceTransformer(model_name)

    # Extraire les questions de la FAQ
    questions = [item["question"] for item in faq_data]

    # Générer les embeddings pour les questions
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Vérification explicite de l'utilisation de NumPy et Obtention de la Dimension des Vecteurs
    if isinstance(embeddings, np.ndarray):
        dimension = embeddings.shape[1]
    else:
        raise ValueError("Les embeddings doivent être un tableau NumPy.")

    return embeddings, model, dimension


def create_faiss_index(embeddings, dimension, faq_data):
    """
    Crée un index FAISS à partir des embeddings générés et stocke les réponses correspondantes.

    Args:
        embeddings (numpy.ndarray): Embeddings des questions de la FAQ.
        dimension (int): Dimension des vecteurs d'embedding.
        faq_data (list): Liste des questions et réponses.

    Returns:
        tuple: Contient l'index FAISS et les réponses correspondantes.
    """
    # Créer un index FAISS utilisant la distance L2 (Euclidienne)
    index = faiss.IndexFlatL2(dimension)

    # Ajouter les embeddings à l'index FAISS
    index.add(embeddings)

    # Stocker les réponses correspondantes
    answers = [item["answer"] for item in faq_data]

    return index, answers

def search_faq(question, index, answers, model, k=1):
    """
    Recherche les réponses les plus pertinentes pour une question donnée en utilisant FAISS.

    Args:
        question (str): La question posée par l'utilisateur.
        index (faiss.Index): L'index FAISS contenant les embeddings des questions.
        answers (list): Liste des réponses correspondantes.
        model (SentenceTransformer): Le modèle d'embedding utilisé.
        k (int): Nombre de réponses à récupérer.

    Returns:
        list: Liste des réponses pertinentes.
    """
    # Générer l'embedding de la question
    query_embedding = model.encode([question], convert_to_numpy=True)

    # Rechercher les k plus proches voisins
    distances, indices = index.search(query_embedding, k)

    # Récupérer les réponses correspondantes
    results = [answers[idx] for idx in indices[0] if idx < len(answers)]

    if results:
        return results
    else:
        return ["Désolé, je n'ai pas trouvé de réponse à votre question."]


def initialize_langchain_model(api_key, model_name="gpt-3.5-turbo"):
    """
    Initialise le modèle de langage LangChain avec OpenAI.

    Args:
        api_key (str): Clé API OpenAI.
        model_name (str): Nom du modèle de langage à utiliser.

    Returns:
        ChatOpenAI: Instance du modèle de langage.
    """
    llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0.7)
    return llm


def generate_faq_response(relavant_docs, question, llm):
    """
    Génère une réponse basée sur les documents pertinents et la question posée en utilisant le modèle de langage LangChain.

    Args:
        relavant_docs (list): Liste des documents pertinents récupérés via FAISS.
        question (str): La question posée par l'utilisateur.
        llm (ChatOpenAI): Instance du modèle de langage.

    Returns:
        str: La réponse générée par le modèle.
    """
    if relavant_docs:
        # Combiner les documents pertinents en contexte
        context = "\n".join(relavant_docs)

        # Créer un prompt avec le contexte et la question
        prompt = f"{context}\n\nQuestion: {question}\nRéponse:"

        # Créer une liste de messages pour le modèle
        messages = [
            SystemMessage(content="Vous êtes un assistant intelligent. Répondez de manière claire et concise."),
            HumanMessage(content=prompt)
        ]

        # Générer la réponse en utilisant le modèle de langage
        response = llm.invoke(messages)

        # Extraire le contenu de la réponse
        answer = response.content
    else:
        answer = "Désolé, je n'ai pas trouvé de réponse à votre question."
    return answer

