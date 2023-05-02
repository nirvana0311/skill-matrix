from flask import Flask, request
from sentence_transformers import SentenceTransformer, util
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import time

# Load pre-trained model bert-base-nli-mean-tokens
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Read CSV file from Azure Blob Storage
connection_string = "DefaultEndpointsProtocol=https;AccountName=documentsourcestorage;AccountKey=ODdUIoI4mP/9mFVgnjZZKzY/9b9EYSGkFwk3QGXoIKlO+w4nYeHIF1rNJt0Z8PRKkeh6ZS4WUySloLv1IbFj3w==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client('cvs')
blob_client = container_client.get_blob_client("skills_pivoted/pivoted_skill_matrix.csv")
downloaded_blob = blob_client.download_blob()
data = downloaded_blob.content_as_text()
pivoted_skill_matrix = pd.read_csv(io.StringIO(data))
skill_list = set(pivoted_skill_matrix.columns)


app = Flask(__name__)

@app.route('/')
def home():
    return 'Please provide a phrase in the URL to shortlist skills with cosine similarity score above 0.75.'

@app.route('/<phrase>')
def shortlist_skills(phrase):
    start_time = time.time()
    # Get embeddings for phrase
    phrase_embeddings = model.encode([phrase])

    # Calculate cosine similarity between phrase and all skills in the skill_list
    cos_sim_list = []
    for skill in skill_list:
        skill_embeddings = model.encode([skill])
        cos_sim = util.cos_sim(phrase_embeddings, skill_embeddings)
        if cos_sim > 0.85:
            cos_sim_list.append((skill, cos_sim))

    # Sort the cos_sim_list in descending order of cosine similarity and select top skills
    top_skills = sorted(cos_sim_list, key=lambda x: x[1], reverse=True)

    # Calculate the time required to serve the request
    end_time = time.time()
    time_taken = end_time - start_time

    # Return the top skills and time taken as a string
    return 'Top skills with cosine similarity score above 0.75 to "{}": {}\nTime taken: {} seconds'.format(phrase, [skill[0] for skill in top_skills], round(time_taken))

if __name__ == '__main__':
    app.run()