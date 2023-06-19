import requests
import os
from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.langchain import LangChain
from github import Github

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    github_url = request.form.get("githubUrl")

    repositories = fetch_user_repositories(github_url)
    if not repositories:
        return jsonify({"error": "No repositories found."})

    most_complex_repository = find_most_complex_repository(repositories)
    if not most_complex_repository:
        return jsonify({"error": "Failed to determine the most complex repository."})

    gpt_analysis = generate_gpt_analysis(most_complex_repository)

    return render_template("result.html", repository=most_complex_repository, analysis=gpt_analysis)
def generate_gpt_analysis(repository):
    # Perform GPT analysis for the repository
    # Implement your logic here

    analysis = "Sample GPT analysis for the repository"
    return analysis


def fetch_user_repositories(github_url):
    api_url = f"https://api.github.com/users/{github_url}/repos"
    response = requests.get(api_url)
    if response.status_code == 200:
        repositories = response.json()
        for repository in repositories:
            repository["url"] = repository["html_url"] 
        return repositories
    else:
        print(f"Error fetching repositories: {response.text}")
        return []

def preprocess_code(code):
    # Implement code preprocessing techniques
    preprocessed_code = code
    return preprocessed_code

def generate_gpt_prompt(repository):
    prompt = f"Analyzing the repository {repository['name']}\n{repository['description']}"
    return prompt

def evaluate_complexity_with_gpt(text):
    # Initialize GPT model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate the output text using the GPT model
    output = model.generate(input_ids)

    # Decode the output tokens to text
    gpt_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return gpt_output

def calculate_complexity_score(text):
    # Initialize LangChain
    langchain = LangChain()

    # Evaluate the technical complexity using LangChain
    complexity_score = langchain.evaluate_complexity(text)

    return complexity_score

def find_most_complex_repository(repositories):
    most_complex_repository = None
    max_complexity_score = float("-inf")

    for repository in repositories:
        processed_code = preprocess_code(repository.get("code",""))
        gpt_prompt = generate_gpt_prompt(repository)
        gpt_output = evaluate_complexity_with_gpt(gpt_prompt)
        complexity_score = calculate_complexity_score(gpt_output)

        if complexity_score > max_complexity_score:
            most_complex_repository = {
                "name": repository["name"],
                "description": repository["description"],
                "url": repository["html_url"]
            }
            max_complexity_score = complexity_score

    return most_complex_repository


if __name__ == "__main__":
    app.run(debug=True)
