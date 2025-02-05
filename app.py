#import pandas as pd
#import numpy as np
#import openai
#import ast
#from openai.embeddings_utils import cosine_similarity

# Load the dataset (assuming it's a CSV)
#df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Preprocess the data (remove punctuation, lowercase, etc.)
#df['Question'] = df['Question'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)

# Initialize OpenAI API key
#openai.api_key =  st.secrets["mykey"]

# Function to get embedding
#def get_embedding(text, model="text-embedding-ada-002"):
#   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Pre-calculate embeddings for all questions in the dataset
#df['Question_Embedding'] = df['Question'].apply(get_embedding)

# Save the DataFrame with embeddings and similarities to a CSV file
#df.to_csv("qa_dataset_with_embeddings.csv", index=False)  # index=False to avoid saving row numbers



# Convert the string embeddings back to lists
#df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

#def find_best_answer(user_question):
   # Get embedding for the user's question
#   user_question_embedding = get_embedding(user_question)

   # Convert to numpy array
#   question_embeddings = np.array(df['Question_Embedding'].tolist())
#   user_embedding = np.array(user_question_embedding).reshape(1, -1)
   
   # Calculate cosine similarities for all questions in the dataset
#   df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

   # Find the most similar question and get its corresponding answer
#   most_similar_index = df['Similarity'].idxmax()
#   max_similarity = df['Similarity'].max()

   # Set a similarity threshold to determine if a question is relevant enough
#   similarity_threshold = 0.6  # You can adjust this value

#   if max_similarity >= similarity_threshold:
#      best_answer = df.loc[most_similar_index, 'Answer']
#      return best_answer
#   else:
#      return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"


# Example Usage
#user_question = "Who will have Cardiomyopathy?"
#best_answer = find_best_answer(user_question)
#print("Best Answer:", best_answer)


import pandas as pd
import numpy as np
import openai
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (assuming it's a CSV)
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Preprocess the data (remove punctuation, lowercase, etc.)
df['Question'] = df['Question'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Initialize OpenAI API key (Replace with your actual key or use an environment variable)
openai.api_key = st.secrets["mykey"]

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        print("Error fetching embedding:", str(e))
        return None

# Ensure embeddings are properly loaded
df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Function to find the best answer
def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    if user_question_embedding is None:
        return "Error generating embedding. Please try again."

    # Convert to numpy array
    question_embeddings = np.array(df['Question_Embedding'].tolist())
    user_embedding = np.array(user_question_embedding).reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(question_embeddings, user_embedding).flatten()

    # Find the most similar question
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[most_similar_index]

    # Set a similarity threshold
    similarity_threshold = 0.6  # Adjust as needed

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask another question?"

# Example Usage
user_question = "Who will have Cardiomyopathy?"
best_answer = find_best_answer(user_question)
print("Best Answer:", best_answer)
