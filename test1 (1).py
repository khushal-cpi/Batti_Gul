import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
hash = {}

df = pd.read_csv('mental_heath_unbanlanced.csv')
my_list  = df['text'].tolist()
embeddings = np.load("dataset_embeddings.npy")
query = 'ironic , he could save others from secrecy-enabled corruption, but not himself'
# print(embeddings)

# query = "I want to kill my enemy " 
# query = input("Enter your R.R. ")

query_embedding = model.encode(query)
cosine_scores = util.cos_sim(query_embedding, embeddings)
order = cosine_scores.tolist()
order[0].sort(reverse = True)
rand = cosine_scores.tolist()

# print(type(cosine_scores.tolist()))

# for i in range(0,5):
#     idx = np.argmax(cosine_scores[i:])
lst = [rand[0].index(order[0][i]) for i in range(0,10)]
print(lst)
for j in lst:
    x = df['status'][j]
    if x in hash:
        hash[x] += 1
    else:
        hash[x] = 1

top = 0
situation = ""
for j in hash:
    if top <= hash[j]:
        situation = j
        top = hash[j]

print(hash)
print(f"You are feeling {situation}")


# embeddings = np.delete(embeddings,idx)




    

# situation = my_list.index(my_list[idx])
# print(f"You are {df['status'][situation]}")

# # while i != 49000:
#     # if df['status'][i] == 'Anxiety':
#     #     anxiety.append(df['text'][i].strip('\n'))
#     # elif df['status'][i] == 'Depression':
#     #     depression.append(df['text'][i].strip('\n'))
#     # elif df['status'][i] == 'Suicidal':
#     #     suicidal.append(df['text'][i].strip('\n'))
#     # elif df['status'][i] == 'Normal':
#     #     normal.append(df['text'][i].strip('\n'))
# # print(df['text'])


# print(normal)

# anxiety_embeddings = model.encode(anxiety)
# # print(anxiety_embeddings)
# depression_embeddings = model.encode(depression)
# suicidal_embeddings = model.encode(suicidal)
# normal_embeddings = model.encode(normal)

# query_string = "I am feeling calm" # Define the query string
# query_embedding = model.encode(query_string) # Encode the query string


# cosine_scores_anxiety = util.cos_sim(query_embedding, anxiety_embeddings)
# cosine_scores_depression = util.cos_sim(query_embedding, depression_embeddings)
# cosine_scores_suicidal = util.cos_sim(query_embedding, suicidal_embeddings)
# cosine_scores_normal = util.cos_sim(query_embedding, normal_embeddings)

# best_idx_anxiety = np.argmax(cosine_scores_anxiety)
# best_idx_depression = np.argmax(cosine_scores_depression)
# best_idx_suicidal = np.argmax(cosine_scores_suicidal)
# best_idx_normal = np.argmax(cosine_scores_normal)

# if max(cosine_scores_anxiety[0][best_idx_anxiety],cosine_scores_depression[0][best_idx_depression],cosine_scores_suicidal[0][best_idx_suicidal],cosine_scores_normal[0][best_idx_normal]) == cosine_scores_anxiety[0][best_idx_anxiety]:
#     situation = "Anxious"
# elif max(cosine_scores_anxiety[0][best_idx_anxiety],cosine_scores_depression[0][best_idx_depression],cosine_scores_suicidal[0][best_idx_suicidal],cosine_scores_normal[0][best_idx_normal]) == cosine_scores_depression[0][best_idx_depression]:
#    situation = "Depressed"
# elif max(cosine_scores_anxiety[0][best_idx_anxiety],cosine_scores_depression[0][best_idx_depression],cosine_scores_suicidal[0][best_idx_suicidal],cosine_scores_normal[0][best_idx_normal]) == cosine_scores_suicidal[0][best_idx_suicidal]:
#    situation = "Suicidal"
# else:
#     situation = "Normal"