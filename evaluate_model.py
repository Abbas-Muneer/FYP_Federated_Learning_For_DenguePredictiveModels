import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes representing different system components
nodes = {
    "Frontend": "Frontend (ReactJS UI)",
    "Backend": "Backend (Flask API)",
    "Database": "Database (SQLite)",
    "Client Upload": "Client Uploads Dataset",
    "Validate Store": "Validate & Store Dataset",
    "Client Train": "Client Local Model Training",
    "FL Train": "Federated Learning Training",
    "Aggregate": "Global Model Aggregation",
    "Evaluate": "Global Model Evaluation",
    "Store Model": "Store & Update Global Model",
    "Download": "Client Downloads Global Model",
    "Display Results": "Display Summary on UI"
}

# Add nodes to the graph
G.add_nodes_from(nodes.keys())

# Define edges (workflow connections between frontend, backend, and database)
edges = [
    ("Frontend", "Client Upload"),  
    ("Client Upload", "Backend"),  
    ("Backend", "Validate Store"),  
    ("Validate Store", "Database"),  

    ("Backend", "Client Train"),  
    ("Client Train", "Database"),  

    ("Frontend", "FL Train"),  
    ("FL Train", "Backend"),  
    ("Backend", "Aggregate"),  
    ("Aggregate", "Evaluate"),  
    ("Evaluate", "Store Model"),  
    ("Store Model", "Database"),  

    ("Frontend", "Download"),  
    ("Download", "Backend"),  
    ("Backend", "Database"),  
    ("Database", "Download"),  

    ("Backend", "Display Results"),  
    ("Database", "Display Results"),  
    ("Display Results", "Frontend")  
]

# Add edges to the graph
G.add_edges_from(edges)

# Define positions for better visualization
pos = {
    "Frontend": (1, 8),
    "Backend": (4, 8),
    "Database": (7, 8),

    "Client Upload": (2, 7),
    "Validate Store": (4, 7),
    "Client Train": (4, 6),

    "FL Train": (2, 5),
    "Aggregate": (4, 5),
    "Evaluate": (6, 5),
    "Store Model": (4, 4),

    "Download": (2, 3),
    "Display Results": (6, 3)
}

# Draw the system architecture graph
plt.figure(figsize=(12, 7))
nx.draw(G, pos, with_labels=True, node_size=3500, node_color="lightblue", font_size=9, font_weight="bold", edge_color="gray", arrows=True)

# Display diagram title
plt.title("System Process Workflow (Frontend, Backend, Database)")
plt.show()
