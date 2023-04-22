import graphviz
import json

# reading json
with open('tree.json', 'r') as f:
    tree_dict = json.load(f)

# tree visualization using graphviz
dot_data = graphviz.Source(tree_dict)
dot_data.view()
dot_data.format = 'png'

