from flask import Flask, render_template, request, session, redirect
import random
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import fitz  # PyMuPDF

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "dev-secret-key"

# Predefined topics for AP Human Geography questions
topics = [
    "Types of maps (reference, thematic)",
    "Spatial concepts (location, place, region, scale, space, connection)",
    "Types of diffusion (relocation, contagious, hierarchical, stimulus)",
    "Types of regions (formal, functional, perceptual)",
    "Geographic data (GIS, GPS, remote sensing)",
    "Population density and distribution",
    "Demographic Transition Model (DTM)",
    "Malthusian theory and critiques",
    "Population policies (pro-natalist, anti-natalist)",
    "Migration types (voluntary, forced, international, internal)",
    "Ravenstein’s Laws of Migration",
    "Push and pull factors",
    "Refugees and internally displaced persons (IDPs)",
    "Folk culture vs. popular culture",
    "Cultural landscape and sequent occupancy",
    "Language families and diffusion",
    "Language extinction and preservation",
    "Universalizing vs. ethnic religions",
    "Religious conflict and sacred spaces",
    "Acculturation, assimilation, syncretism",
    "Ethnic enclaves and cultural hearths",
    "States, nations, nation-states, stateless nations",
    "Sovereignty and territoriality",
    "Colonialism and imperialism",
    "Types of boundaries and boundary disputes",
    "Gerrymandering and redistricting",
    "Supranational organizations (UN, EU, NATO)",
    "Centripetal and centrifugal forces",
    "Devolution and independence movements",
    "First, Second, and Green Agricultural Revolutions",
    "Subsistence vs. commercial agriculture",
    "Agricultural regions and practices",
    "Land survey systems (metes and bounds, long lot, township and range)",
    "Rural settlement patterns",
    "Agribusiness and monoculture",
    "Food deserts and food security",
    "Desertification and sustainability in agriculture",
    "Urban hierarchy and functional zonation",
    "Gentrification and urban renewal",
    "Suburbanization and smart growth",
    "Economic sectors (primary, secondary, tertiary, quaternary, quinary)",
    "Measures of development (GDP, GNI, HDI)",
    "Rostow’s Stages of Economic Growth",
    "Wallerstein’s World Systems Theory",
    "Core-periphery and dependency theories",
    "Outsourcing, offshoring, and globalization",
    "Special Economic Zones (SEZs) and trade policies",
    "Fair trade, sustainable development, and microfinance"
]


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def create_vector_store(text):
    """Creates a vector store from the extracted text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_texts(texts, embeddings)


def search_relevant_info(topic, vector_store, k=3):
    """Finds the most relevant information in the vector store for the given topic."""
    return vector_store.similarity_search(topic, k=k)


def generate_question(topic, context):
    """Generates an AP Human Geography question based on a topic and context."""
    llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    prompt = PromptTemplate.from_template(
        "Based on the following context,\n\n"
        "Topic: {topic}\n"
        "Context: {context}\n\n"
        "Create An AP Human Geography short answer question formatted as where questions are related\n"
        "a) \n"
        "b) \n"
        "c) \n"
        "d) \n"
        "e) \n"
        "f) \n"
        "With no additional text"
    )
    return llm.invoke(prompt.format(topic=topic, context=context))


def grade_response(student_response, question_part, context):
    """
    Grades the student's response for a specific question part.
    Returns feedback with score (0 or 1).
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=api_key)  # Lower temperature for more consistent results

    prompt = f"""
    You are grading an AP Human Geography response. Evaluate strictly based on these rules:

    1. Score 1 ONLY if the response:
       - Directly answers the SPECIFIC question part asked
       - Contains accurate, relevant information
       - Demonstrates understanding of the concept
    2. Score 0 if:
       - Response is incorrect or irrelevant
       - Doesn't address the specific question part
       - Contains random characters/nonsense

    QUESTION PART: {question_part}
    STUDENT RESPONSE: {student_response}
    RELEVANT CONTEXT: {context}

    Provide feedback that:
    - Clearly states if the answer was correct/incorrect
    - Explains WHY it was correct/incorrect
    - References the specific question part
    - Does NOT include generic advice

    Format EXACTLY like this:
    FEEDBACK: [concise feedback focusing on this specific question]
    SCORE: [0 or 1]
    """

    result = llm.invoke(prompt)
    feedback = result.content

    # Ensure consistent formatting
    if "FEEDBACK:" not in feedback:
        feedback = f"FEEDBACK: {feedback}"
    if "SCORE:" not in feedback:
        feedback += "\nSCORE: 0"  # Default to 0 if score missing

    return feedback


pdf_text = extract_text_from_pdf("ap_human_geo.pdf")
vector_store = create_vector_store(pdf_text)


def get_random_topic_and_info():
    topic = random.choice(topics)
    relevant_info = search_relevant_info(topic, vector_store)
    return topic, relevant_info[0].page_content


def parse_question_parts(full_question):
    """Parses the full question into individual parts."""
    parts = {}
    current_part = None
    for line in full_question.split('\n'):
        if line.strip().startswith(('a)', 'b)', 'c)', 'd)', 'e)', 'f)')):
            current_part = line.strip()[0]
            parts[current_part] = line.strip()[2:].strip()
        elif current_part:
            parts[current_part] += " " + line.strip()
    return parts


def parse_score(feedback):
    """Extracts score from feedback text."""
    if "SCORE: 1" in feedback:
        return 1
    return 0

@app.route("/home")
def welcome():
    """Renders the welcome/home screen"""
    return render_template("welcome.html")

# This should come right before your existing main route
@app.route("/")
def root():
    """Redirects to the home screen"""
    return redirect("/home")
@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if request.method == "GET" or 'current_question' not in session:
        topic, relevant_info = get_random_topic_and_info()
        full_question = generate_question(topic, relevant_info).content
        question_parts = parse_question_parts(full_question)

        session['current_question'] = question_parts
        session['current_topic'] = topic
        session['current_context'] = relevant_info

        return render_template("index.html",
                               topic=topic,
                               question_parts=question_parts)

    if request.method == "POST":
        question_parts = session['current_question']
        topic = session['current_topic']
        context = session['current_context']

        responses = {}
        feedbacks = {}
        scores = {}
        total_score = 0

        for part in question_parts:
            response = request.form.get(f"response_{part}", "")
            responses[part] = response
            feedback = grade_response(response, question_parts[part], context)
            feedbacks[part] = feedback
            score = parse_score(feedback)
            scores[part] = score
            total_score += score

        return render_template("result.html",
                               total_score=total_score,
                               individual_scores=scores,
                               feedbacks=feedbacks,
                               topic=topic,
                               question_parts=question_parts,
                               responses=responses)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)


