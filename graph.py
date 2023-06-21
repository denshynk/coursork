import networkx as nx
import random
import matplotlib.pyplot as plt
import time

def generate_graph(n):
    # Создаем пустой граф
    G = nx.Graph()

    # Устанавливаем количество узлов, соединенных с A и B
    n_start = random.randint(2, 4)  # Количество узлов, соединенных с A
    n_end = random.randint(2, 4)  # Количество узлов, соединенных с B

    # Добавляем узлы в граф и назначаем им случайные цены
    for i in range(n):
        vertex_price = random.randint(10, 59)  # Случайная цена от 10 до 59
        nx.set_node_attributes(G, {i: vertex_price}, 'vertex_price')


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


def dijkstra_shortest_path(graph, source, target):
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


def greedy(G, hotel_prices):
    lowest_cost = float("inf")
    result_city = -1
    shortest_path = []
    for i in range(1, G.number_of_nodes() - 1):
        total_cost = 0
        try:
            a_path, a_path_cost = dijkstra_shortest_path(G, "A", i)
        except nx.NodeNotFound:
            a_path_cost = float("inf")
        try:
            b_path, b_path_cost = dijkstra_shortest_path(G, "B", i)
        except nx.NodeNotFound:
            b_path_cost = float("inf")

        if len(a_path) == len(b_path):
            total_cost = a_path_cost + b_path_cost
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                result_city = i
                shortest_path = a_path + b_path[::-1][1:]

        else:
            total_cost = a_path_cost + b_path_cost + hotel_prices[i - 1] * abs(len(a_path) - len(b_path))
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                result_city = i
                shortest_path = a_path + b_path[::-1][1:]

        print(f"Shortest path from A to {i} is {a_path}")
        print(f"Shortest path from B to {i} is {b_path}")
        print(f"Current cost is {total_cost}")
        print(f"Lowest cost is {lowest_cost}\n")

    return lowest_cost, result_city, shortest_path


# Number of graphs to generate
num_graphs = 5

total_execution_time = 0.0

# Generate and display graphs
for i in range(num_graphs):
    G = generate_graph(n=20)  # Set the number of vertices for each graph
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

    # Execute greedy algorithm to find the shortest path from vertex A to vertex B
    hotel_prices = [G.nodes[node]['vertex_price'] for node in list(G.nodes())[1:-1]]  # Get hotel prices for greedy function


    start_time = time.time()
    lowest_cost, result_city, shortest_path = greedy(G, hotel_prices)
    end_time = time.time()

    execution_time = end_time - start_time
    total_execution_time += execution_time

    print(f"\nLowest cost: {lowest_cost}")
    print(f"Result city: {result_city}")
    print(f"Shortest path: {shortest_path}")
    print(f"Execution time: {execution_time} seconds\n")

print(f"Total execution time for {num_graphs} graphs: {total_execution_time} seconds")
