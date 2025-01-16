import os
import re

from msglite import Message
from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from warnings import simplefilter

if __name__ == '__main__':
    # Ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    bodies = []
    cleared_bodies = []
    # Set number of clusters
    true_k = 10
    # Opening .msg files and reading message bodies
    for filename in os.listdir('emails'):
        if filename.endswith('.msg'):
            msg = Message(os.path.join('emails', filename))
            bodies.append(msg.body)
        elif filename.endswith('.eml'):
            with open(os.path.join('emails', filename), 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
                bodies.append(msg.as_string())
    # Clearing of bodies
    for body in bodies:
        body = re.sub(r'<[^>]+>', '', body, flags=re.S)
        body = body.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        body = body.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').replace('-', ' ')
        body = body.replace('<', ' ').replace('>', ' ').replace('  ', ' ')
        cleared_bodies.append(body)
    # Convert bodies to vectors
    vectorizer = TfidfVectorizer(lowercase=True,
                                 max_features=100,
                                 max_df=5,
                                 min_df=0.8,
                                 ngram_range=(1, 3),
                                 stop_words='english')
    vectors = vectorizer.fit_transform(cleared_bodies)
    # Creating dictionary of similar emails
    similarity_dict = {}
    for i in range(len(cleared_bodies) - 1):
        max_similarity = 0

        for j in range(1, len(cleared_bodies)):
            similarity = float(cosine_similarity(vectors[i], vectors[j]))
            print(i, j)
            print(cleared_bodies[i])
            print(cleared_bodies[j])
            print(similarity)
            if 1 > similarity > max_similarity and cleared_bodies[i] != cleared_bodies[j]:
                max_similarity = similarity
                similarity_dict.update({max_similarity: [i, j]})
    # Sorting list of dictionary keys(by similarity from biggest to lowest)
    similarity_list = sorted(similarity_dict.keys(), reverse=True)
    # Creating output file, clusters is distributed by similarity
    with open('results.txt', 'w', encoding='utf-8') as f:
        for i in range(true_k):
            f.write(f'Cluster {i}')
            f.write('\n')

            for j in similarity_list:
                if 1 - i / 10 >= j >= 1 - (i + 1) / 10:
                    messages = similarity_dict[j]
                    f.write(cleared_bodies[messages[0]])
                    f.write('\n')
                    f.write('-' * 128)
                    f.write('\n')
                    f.write(cleared_bodies[messages[1]])
                    f.write('\n')
                    f.write('-' * 128)
                    f.write('\n')
                    f.write('Cosine similarity = ' + str(j))
                    f.write('\n')
                    f.write('*' * 128)
                    f.write('\n')
            f.write('\n')
