## Main goal:
# 1. Identifying the most important characters?
# 2. Examining how the importance of these characters changes over time?
# 3. Identifying the main communities within the network?
# 4. subcommunity/clusters sta ih veze

import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from pyvis.network import Network
import community as community_louvain
from lib.utils.functions import *

# Load English tokenizer, tagger, parser and NER
nlp=spacy.load('en_core_web_sm')

#TODO From book create list of sentences
#TODO Make list of all characters that are mentioned in each sentence
#TODO Mention window - if 2 (or more) characters are mentioned in this windows then they will be in relation
#TODO names recognition

############################* Load books ############################

# Get all book files in the data directory
all_books = [book for book in os.scandir('data') if '.txt' in book.name]
# print(all_books)

# Get text from book
book1 = all_books[1]
book1_text = open(book1).read()
# print(book1_text)
book_doc = nlp(book1_text)


# ############################* Visualize identified entities with displacy and show them as HTML ############################

# # Visualize identified entities
# displacy_data = displacy.render(book_doc[2000:3000], style='ent', jupyter=False, options={'distance': 100})

# # Save HTML markup to a file
# output_file = 'displacy_output.html'
# with open(output_file, 'w', encoding='utf-8') as file:
#     file.write(displacy_data)

# # Open the saved HTML file in a web browser
# os.system(f'start {output_file}')


##* Load character DataFrame with names
character_df = pd.read_csv("all_characters.csv")
#print(character_df.to_string(index=False))

# Remove brackets and text within brackets
character_df['character'] = character_df["character"].apply(lambda x: re.sub(r"[\(].*?[\)]", "", x))

# Create First name because some characters are mentioned like this "Geralt of The Rivia"
character_df['character_firstname'] = character_df["character"].apply(lambda x: x.split(' ', 1)[0])

pd.set_option('display.max_rows', None)
# print(character_df)


############################* Get names from each sentences ############################

sent_names_df = []

# Loop through sentences, store names list for each sentence
for sentence in book_doc.sents:
    name_list = [name.text for name in sentence.ents]
    sent_names_df.append({"sentence": sentence, "names": name_list})

sent_names_df = pd.DataFrame(sent_names_df)
# print(sent_names_df)
sent_names_df.to_csv('sent_names.csv', index=False)
# print(sent_names_df.to_string(index=False))

# Function to filter out non-character names
def filter_names(name_list, character_df):
    return [name for name in name_list 
            if name in list(character_df.character)
            or name in list(character_df.character_firstname)]
# print(filter_names(["Geralt", "ankica", "Thu", "2"], character_df))

sent_names_df["character_names"] = sent_names_df['names'].apply(lambda x: filter_names(x, character_df))
# print(sent_names_df.head(10))

# Filter out sentences that don't have any character
sent_names_df_filtered = sent_names_df[sent_names_df["character_names"].map(len) > 0]
# print(sent_names_df_filtered.head(10))

# Take only first name of character and not whole name (like Geralt of Rivia)
sent_names_df_filtered["character_names"] = sent_names_df_filtered["character_names"].apply(lambda x: [name.split()[0] for name in x])
pd.reset_option('^display.', silent= True)
# print(sent_names_df_filtered)


############################* Create relationships ############################

# Determine the distance size of sentences for names to be in relationship
window_size = 5
relationships = []

for i in range(sent_names_df_filtered.index[-1]): # -->  [-1] from last index
    end_i = min(i + 5, sent_names_df_filtered.index[-1]) # --> we use min so we dont exceed index when we get to the last 5 rows of DataFrame
    char_list = sum((sent_names_df_filtered.loc[i: end_i].character_names), []) # --> all names that appear in window_size rows

    # Remove duplicated characters that are next to each other
    char_unique = [char_list[i] for i in range(len(char_list))
                   if (i==0) or char_list[i] != char_list[i-1]]
    

    if len(char_unique) > 1: # --> if there is only one character there is no relationship
        for idx, a in enumerate(char_unique[:-1]): # --> iterate to second last character in list, so that we dont exceed index value
            b = char_unique[idx+1]
            relationships.append({
                "source": a,
                "target": b,
                # "distance": , #todo --> think how can you define distance 
            })

relationships_df = pd.DataFrame(relationships) 
pd.set_option('display.max_rows', None)
# print(relationships_df.head(50))

# Aggregate relationships (Ciri -> Geralt is same as Geralt -> Ciri), sort the cases with a->b and b->a
relationships_df = pd.DataFrame(np.sort(relationships_df.values, axis=1), columns= relationships_df.columns)

# Create values/weight for each relationship
relationships_df["value"] = 1
relationships_df = relationships_df.groupby(["source", "target"], sort=False, as_index=False).sum()
# print(relationships_df.head(10))


############################* Graph analysis and visualization ############################

# Create a networkx graph from the dataframe
G = nx.from_pandas_edgelist(relationships_df,
                            source="source",
                            target="target",
                            edge_attr="value",
                            create_using=nx.Graph())

# Get the number of nodes and edges in the graph
print("Number of nodes: {}".format(G.number_of_nodes()))
print("Number of edges: {}".format(G.number_of_edges()))

# Graph visualization - Network X
plt.figure(figsize=(10,10))
pos = nx.kamada_kawai_layout(G)
nx.draw(G, with_labels = True, node_color = 'skyblue', edge_cmap = plt.cm.Blues, pos = pos)
plt.show()

# Graph visualization - Pyvis
net = Network(height='100dvh', width='100%', bgcolor='#222222', font_color='white')

# Make nodes size based on "degree", which are connections between nodes,
# This is made based on Degree centrality... more about this below
node_degree = dict(G.degree)

# Setting up node size attribute
nx.set_node_attributes(G, node_degree, 'size')

net.from_nx(G)
net.show('index.html', notebook = False) # --> need to add notebook = False to work!!!

## Degree centrality
# Degree centrality is a measure of the relative importance of a node in a network based on its degree, which is the number of edges incident to the node. In simple terms, it measures how well connected a node is within the network. Nodes with higher degree centrality are considered more central or influential in the network, as they have more connections to other nodes. They often play important roles in information flow, communication, and influence within the network.

## Closeness Centrality
# Closeness centrality measures how close a node is to all other nodes in the network. It is defined as the reciprocal of the sum of the shortest path distances between a node and all other nodes in the network. In other words, it quantifies how quickly a node can interact with other nodes in the network. Nodes with higher closeness centrality are those that are on average closer to all other nodes and can more efficiently communicate or access information in the network.

## Betweenness Centrality
# Betweenness centrality measures the extent to which a node lies on the shortest paths between pairs of other nodes in the network. It quantifies how often a node acts as a bridge along the shortest path between two other nodes. Nodes with higher betweenness centrality are those that are crucial for maintaining connectivity in the network, as they control the flow of information between different parts of the network. They often serve as key intermediaries or bottlenecks in the network.

## Eigenvector Centrality
# Eigenvector centrality measures the influence of a node in the network based on the concept of eigenvectors. It assigns a centrality score to each node proportional to the sum of the centrality scores of its neighboring nodes, with the weights of the edges taken into account. In other words, nodes with higher eigenvector centrality are those that are connected to other highly central nodes. It reflects the idea that a node is important if it is connected to other important nodes. This centrality measure is often used to identify influential or prestigious nodes in the network.


############################* The most important characters in The Witcher ############################

# Degree centrality
degree_dict = nx.degree_centrality(G)
# print(degree_dict)
degree_df = pd.DataFrame.from_dict(degree_dict, orient="index", columns=["centrality"])
# Plot top 10 nodes
centr_deg_plt =degree_df.sort_values("centrality", ascending=False)[0:9].plot(kind="bar")
plt.show()

# Betweenness centrality
betweenness_dict = nx.betweenness_centrality(G)
# print(betweenness_dict)
betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient="index", columns=["betweenness"])
# Plot top 10 nodes
betweenness_deg_plt =betweenness_df.sort_values("betweenness", ascending=False)[0:9].plot(kind="bar")
plt.show()

# Closeness centrality
closeness_dict = nx.closeness_centrality(G)
closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['closeness'])
# Plot top 10 nodes
closeness_deg_plt = closeness_df.sort_values('closeness', ascending=False)[0:9].plot(kind="bar")
plt.show()

# Save centrality measures
nx.set_node_attributes(G, degree_dict, 'degree_centrality')
nx.set_node_attributes(G, betweenness_dict, 'betweenness_centrality')
nx.set_node_attributes(G, closeness_dict, 'closeness_centrality')


############################* Community detection ############################

# Community detection
communities = community_louvain.best_partition(G)
nx.set_node_attributes(G, communities, 'group')

# Create visualization with Pyvis
com_net = Network(width="100%", height="100dvh", bgcolor='#222222', font_color='white')
com_net.from_nx(G)
com_net.show("witcher_communities.html", notebook = False) # --> need to add notebook = False to work!!!


############################* Evolution of characters' importance ############################
#! Need time to complete this task... leave PC for about 5-10min

# Initialize empty list for graphs from books
books_graph = []

# Sort dir entries by name
all_books.sort(key=lambda x: x.name)

# Loop through book list and create graphs
for book in all_books:
    book_text = ner(book)
    
    # Get list of entities per sentences
    sent_entity_df = get_ne_list_per_sentence(book_text)
    
    # Select only character entities
    sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, character_df))

    # Filter out sentences that don't have any character entities
    sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]
    
    # Take only first name of characters
    sent_entity_df_filtered['character_entities'] = sent_entity_df_filtered['character_entities'].apply(lambda x: [item.split()[0] 
                                                                                                               for item in x])

    # Create relationship df
    relationship_df = create_relationships(df = sent_entity_df_filtered, window_size = 5)                                                                                                               
    
    # Create a graph from a pandas dataframe
    G = nx.from_pandas_edgelist(relationship_df, 
                                source = "source", 
                                target = "target", 
                                edge_attr = "value", 
                                create_using = nx.Graph())     
    
    books_graph.append(G) 


# Creating a list of degree centrality of all the books
evol = [nx.degree_centrality(book) for book in books_graph]

# Creating a DataFrame from the list of degree centralities in all the books
degree_evol_df = pd.DataFrame.from_records(evol)

# Plotting the degree centrality evolution of 5 main characters
degree_evol_df[["Geralt", "Ciri", "Yennefer", "Dandelion", "Vesemir"]].plot()
plt.show()