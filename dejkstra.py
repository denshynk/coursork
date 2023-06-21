import networkx as nx
import random
import matplotlib.pyplot as plt
import time

def generate_graph(n):
    # Create an empty graph
    G = nx.Graph()

    # Set the number of vertices connected to A and B
    n_start = random.randint(2, 4)  # Number of vertices connected to A
    n_end = random.randint(2, 4)  # Number of vertices connected to B

    # Add vertices to the graph and assign random prices to them
    for i in range(n):
        vertex_price = random.randint(10, 59)  # Random price from 10 to 59
        G.add_node(vertex_price)

    # Add random edges between vertices with random costs
    for i, vertex in enumerate(list(G.nodes())):
        # Add a random number of edges between the current vertex and other vertices
        if i < n_start:
            num_edges = random.randint(2, 4)  # Number of edges from 2 to 4 for A
        elif i >= n - n_end:
            num_edges = random.randint(2, 4)  # Number of edges from 2 to 4 for B
        else:
            num_edges = random.randint(2, 4)  # Number of edges from 1 to 3 for other vertices

        for _ in range(num_edges):
            random_vertex = random.choice(list(G.nodes()))
            # Check that the edge does not connect a vertex to itself
            if random_vertex != vertex and not G.has_edge(vertex, random_vertex):
                edge_cost = random.randint(10, 59)  # Random edge cost from 10 to 59
                G.add_edge(vertex, random_vertex, cost=edge_cost)

    # Add the start vertex A and connect it to 2-4 first vertices
    start_vertex = "A"
    G.add_node(start_vertex)

    for i in range(n_start):
        random_vertex = random.choice(list(G.nodes()))
        # Check that the edge does not connect the vertex to itself
        if random_vertex != start_vertex:
            edge_cost = random.randint(10, 59)  # Random edge cost from 10 to 59
            G.add_edge(start_vertex, random_vertex, cost=edge_cost)

    # Add the end vertex B and connect it to 2-4 last vertices
    end_vertex = "B"
    G.add_node(end_vertex)

    for i in range(n_end):
        random_vertex = random.choice(list(G.nodes()))
        # Check that the edge does not connect the vertex to itself
        if random_vertex != end_vertex:
            edge_cost = random.randint(10, 59)  # Random edge cost from 10 to 59
            G.add_edge(end_vertex, random_vertex, cost=edge_cost)

    # Remove common edges between A and B if they exist
    if G.has_edge(start_vertex, end_vertex):
        G.remove_edge(start_vertex, end_vertex)

    return G


def dijkstra_shortest_path(graph):
    source = "A"
    target = "B"
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0

    visited = set()

    while len(visited) < len(graph.nodes()):
        current_node = min((node for node in graph.nodes() if node not in visited), key=distances.get)
        visited.add(current_node)

        for neighbor in graph.neighbors(current_node):
            cost = graph.get_edge_data(current_node, neighbor)['cost']
            new_distance = distances[current_node] + cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance

    shortest_path = [target]
    current_node = target
    while current_node != source:
        neighbors = list(graph.neighbors(current_node))
        min_neighbor = min(neighbors, key=distances.get)
        shortest_path.append(min_neighbor)
        current_node = min_neighbor

    shortest_path.reverse()

    return shortest_path


# Number of graphs to generate
num_graphs = 1000

total_execution_time = 0.0

# Generate and display graphs
for i in range(num_graphs):
    G = generate_graph(n=10000)  # Set the number of vertices for each graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)

    # Get price labels for vertices
    labels = nx.get_node_attributes(G, 'price')

    # Display price labels on vertices
    nx.draw_networkx_labels(G, pos, labels=labels)

    # Display edge cost labels
    edge_labels = nx.get_edge_attributes(G, 'cost')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    #plt.show()

    # Execute Dijkstra's algorithm to find the shortest path from vertex A to vertex B
    

    start_time = time.time()
    shortest_path = dijkstra_shortest_path(G)
    end_time = time.time()

    execution_time = (end_time - start_time)*1000
    total_execution_time += execution_time

    print(f"Shortest path from A to B:")
    print(shortest_path)

    # Calculate the sum of edge costs in the path
    path_sum = sum(G.get_edge_data(shortest_path[i], shortest_path[i + 1])['cost'] for i in range(len(shortest_path) - 1))

    # Check if the path length is even and add the smaller central element if it is
    if len(shortest_path) % 2 == 0:
        middle_index1 = len(shortest_path) // 2 - 1
        middle_index2 = len(shortest_path) // 2
        central_elements = [shortest_path[middle_index1], shortest_path[middle_index2]]
        smallest_central_element = min(central_elements)
        path_sum += smallest_central_element
        print(f"City to Stop for whait: {smallest_central_element}")

    print(f"Sum of edge costs in the path: {path_sum}")
    #plt.show()
print(f"Total execution time for {num_graphs} graphs: {total_execution_time} seconds")
