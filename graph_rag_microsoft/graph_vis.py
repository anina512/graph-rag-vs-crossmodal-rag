import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

# Load GraphML file
G = nx.read_graphml(r"C:\Internship\graph_rag_GM\output\graph.graphml")

# Create PyVis network
net = Network(
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#222222"
)

# Enable physics (forces system)
net.barnes_hut(
    gravity=-8000,
    central_gravity=0.2,
    spring_length=150,
    spring_strength=0.02,
    damping=0.09,
)

# -----------------------------
#  COMMUNITY DETECTION
# -----------------------------
communities = list(greedy_modularity_communities(G))
community_map = {}

for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

# -----------------------------
#  Add nodes
# -----------------------------
for node in G.nodes():
    deg = G.degree(node)
    group = community_map.get(node, 0)

    net.add_node(
        node,
        label=node,
        title=f"Node: {node}<br>Degree: {deg}",
        size=10 + deg * 3,
        group=group
    )

# -----------------------------
#  Add edges
# -----------------------------
for source, target in G.edges():
    net.add_edge(source, target)

# -----------------------------
#  VALID JSON OPTIONS (FIXED)
# -----------------------------
options_json = """
{
  "nodes": {
    "shape": "dot",
    "borderWidth": 1,
    "font": { "size": 20 }
  },
  "edges": {
    "color": { "color": "#666666", "highlight": "#333333" },
    "smooth": false
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -8000,
      "springLength": 175,
      "springConstant": 0.015,
      "damping": 0.25,
      "avoidOverlap": 1
    },
    "stabilization": { "iterations": 250 }
  }
}
"""

net.set_options(options_json)

#net.toggle_physics(False)

# Output
net.show("interactive_graph.html", notebook=False)

print("Graph generated! Open 'interactive_graph.html'.")