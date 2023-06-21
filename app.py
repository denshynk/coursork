import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
from itertools import islice
from timeit import default_timer as timer
import msvcrt

def menu():
    while True:
        print("Меню:")
        print("1. Ввести данні самостійно")
        print("2. Зчитати данні з файлу")
        print("3. Згенерувати випадково")
        print("4. Провести експеримент")
        print("0. Завершити роботу")

        choice = input("Виберіть пункт меню: ")

        if choice == "1":
            enter_data_manually()
        elif choice == "2":
            enter_data_file()
        elif choice == "3":
            generate_data()
        elif choice == "4":
            conduct_experiment()
        elif choice == "0":
            print("Завершення роботи.")
            break
        else:
            print("Некоректний вибір. Будь ласка, виберіть пункт меню знову.")

def second_menu(G):
    while True:
        print("Підменю Післяведення даних:")
        print("1. Розв'язати задачу всіма розробленими алгоритмами")
        print("2. Розв'язати задачу алгоритмом Дейкстри")
        print("3. Розв'язати задачу жадібним алгоритмом")
        print("4. Розв'язати задачу жадібним модифікованим алгоритмом")
        print("0. Завершити роботу")

        choice = input("Виберіть пункт підменю: ")

        if choice == "1":
            solve_task(G)
        elif choice == "2":
            print()
            solve_dijkstra(G)
        elif choice == "3":
            print()
            solve_greedy(G)
        elif choice == "4":
            print()
            solve_greedy_modified(G)
        elif choice == "0":
            break
        else:
            print("Некоректний вибір. Будь ласка, виберіть пункт підменю знову.")


def enter_data_manually():
    print("Введення даних задачі самостійно")
    G = nx.Graph()

    # Ввод количества вершин в графе
    num_vertices = int(input("Введите количество вершин в графе: "))

    # Ввод вершин и их весов
    for i in range(num_vertices):
        i += 1
        vertex = str(i)
        cost = int(input(f"Введите вес для вершины {vertex}: "))
        G.add_node(vertex, cost=cost)

    # Автоматическое добавление вершин A и B
    G.add_node('A')
    G.add_node('B')

    # Ввод ребер и их весов
    while True:
        edge_input = input("Введите ребро (формат: вершина1 вершина2, или оставьте пустым для завершения): ")
        if not edge_input:
            break

        u, v = edge_input.split()
        # Проверка, что вершины u и v существуют в графе
        if u in G.nodes() and v in G.nodes():
            edge_cost = int((input(f"Введите вес для ребра ({u}, {v}): ")))
            G.add_edge(u, v, cost=edge_cost)
        else:
            print("Ошибка: одна или обе вершины не существуют в графе.")

    return second_menu(G)

def enter_data_file():
    print("Зчитування даних задачі з файлу")
    G = nx.Graph()

    filename = input("Введіть ім'я файлу: ")
    try:
        with open(filename, 'r') as file:
            num_vertices = int(file.readline().strip())

            for i in range(num_vertices):
                i += 1
                vertex, cost = file.readline().strip().split()
                G.add_node(vertex, cost=int(cost))

            G.add_node('A')
            G.add_node('B')

            while True:
                edge_input = file.readline().strip()
                if not edge_input:
                    break

                u, v, edge_cost = edge_input.split()
                if u in G.nodes() and v in G.nodes():
                    G.add_edge(u, v, cost=int(edge_cost))
                else:
                    print("Помилка: одна або обидві вершини не існують у графі.")

        return second_menu(G)

    except FileNotFoundError:
        print("Помилка: файл не знайдено.")
        return
   
def generate_data():
    n = int(input("Введите количество вершин в графе: "))
    # Создаем пустой граф
    G = nx.Graph()

    # Задаем количество вершин, соединенных с вершиной A и B
    n_start = random.randint(2, 4)  # Количество вершин, соединенных с A
    n_end = random.randint(2, 4)  # Количество вершин, соединенных с B

    # Добавляем вершины в граф и присваиваем им случайные цены
    user_input = input("Введите диапазон чисел для установки стоимости проживания (например, 10 59): ")
    if user_input.strip() == "":
        min_value = 10
        max_value = 59
    else:
        min_value, max_value = map(int, user_input.split())
    for i in range(n):
        vertex_price = random.randint(min_value, max_value)
        G.add_node(i, cost = vertex_price)

    # Добавляем случайные ребра между вершинами с случайными стоимостями
    user_input_way = input("Введите диапазон чисел для установки стоимости передвижения (например, 10 59): ")
    if user_input.strip() == "":
        min_value_way = 10
        max_value_way = 59
    else:
        min_value_way, max_value_way = map(int, user_input_way.split())
    for i, vertex in enumerate(list(G.nodes())):
        # Добавление случайного количества ребер между текущей вершиной и другими вершинами
        if i < n_start:
            num_edges = random.randint(2, 4)  # Количество ребер от 2 до 4 для A
        elif i >= n - n_end:
            num_edges = random.randint(2, 4)  # Количество ребер от 2 до 4 для B
        else:
            num_edges = random.randint(2, 4)  # Количество ребер от 1 до 3 для остальных вершин

        for _ in range(num_edges):
            random_vertex = random.choice(list(G.nodes()))
            # Проверка, чтобы ребро не соединяло вершину с самой собой
            if random_vertex != vertex and not G.has_edge(vertex, random_vertex):
                edge_cost = random.randint(min_value_way, max_value_way)  # Случайная стоимость ребра от 10 до 59
                G.add_edge(vertex, random_vertex, cost=edge_cost)
                

    # Добавляем начальную вершину A и соединяем ее с 2-4 первыми вершинами
    start_vertex = 'A'
    G.add_node(start_vertex)

    for i in range(n_start):
        random_vertex = random.choice(list(G.nodes()))
        # Проверка, чтобы ребро не соединяло вершину с самой собой
        if random_vertex != start_vertex:   
            edge_cost = random.randint(min_value_way, max_value_way)  # Случайная стоимость ребра от 10 до 59
            G.add_edge(start_vertex, random_vertex, cost=edge_cost)

    # Добавляем конечную вершину B и соединяем ее с 2-4 последними вершинами
    end_vertex = 'B'
    G.add_node(end_vertex)

    for i in range(n_end):
        random_vertex = random.choice(list(G.nodes()))
        # Проверка, чтобы ребро не соединяло вершину с самой собой
        if random_vertex != end_vertex:
            edge_cost = random.randint(min_value_way, max_value_way)  # Случайная стоимость ребра от 10 до 59
            G.add_edge(end_vertex, random_vertex, cost=edge_cost)

    # Удаляем общие ребра между A и B, если они существуют
    if G.has_edge(start_vertex, end_vertex):
        G.remove_edge(start_vertex, end_vertex)

    return second_menu(G)

def dijkstra_shortest_path(graph):
        source = 'A'
        target = 'B'
        distances = {node: float('inf') for node in graph.nodes()}
        distances[source] = 0

        visited = set()

        while len(visited) < len(graph.nodes()):
            current_node = min((node for node in graph.nodes() if node not in visited), key=distances.get)
            visited.add(current_node)

            for neighbor in graph.neighbors(current_node):
                cost = int(graph.get_edge_data(current_node, neighbor)['cost'])
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
        print("Рішення за допомогою Алгоритму Дейкстри")
        print("Shortest path between A and B:")
        print(shortest_path)

        path_sum = sum(int(graph.get_edge_data(shortest_path[i], shortest_path[i + 1])['cost']) for i in range(len(shortest_path) - 1))
        if len(shortest_path) % 2 == 0:
            middle_index1 = len(shortest_path) // 2 - 1
            middle_index2 = len(shortest_path) // 2
            central_elements = [shortest_path[middle_index1], shortest_path[middle_index2]]
            smallest_central_element = min(central_elements)
            path_sum += int(smallest_central_element)
            print(f"City to Stop for wait: {smallest_central_element}")
        ##else:
            
            
        print(f"Sum of edge costs in the path: {path_sum}")

        return shortest_path

# Greedy algorithms
def k_shortest_paths(G, source, target, k, weight = "cost"):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def find_shortest_path_of_length(paths, length):
	if paths is not None:
		for path in paths:
			if len(path) == length:
				return path
	return None

def greedy(G, weightKey = "cost"):
    print('Greedy Algorithm: ')
    lowest_cost = sys.maxsize
    result_city = -1
    shortest_path = []
    hotel_prices = nx.get_node_attributes(G, weightKey)
    for current_node in G.nodes:
        if current_node != 'A' and current_node != 'B':
            total_cost = 0
            a_path = nx.dijkstra_path(G, 'A', current_node, weight = weightKey)
            b_path = nx.dijkstra_path(G, 'B', current_node, weight = weightKey)
            a_path_cost = nx.path_weight(G, a_path, weight = weightKey)
            b_path_cost = nx.path_weight(G, b_path, weight = weightKey)
            if len(a_path) == len(b_path):
                total_cost = a_path_cost + b_path_cost
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    result_city = current_node
                    t = b_path.copy()
                    t.pop()
                    t.reverse()
                    shortest_path = a_path + t
            else:
                total_cost = a_path_cost + b_path_cost + hotel_prices[current_node] * abs(len(a_path) - len(b_path))
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    result_city = current_node
                    t = b_path.copy()
                    t.pop()
                    t.reverse()
                    shortest_path = a_path + t
            # print("Shortest path from A to ", current_node, "is ", a_path)
            # print("Shortest path from B to ", current_node, "is ", b_path)
            # print("Current cost is ", total_cost)
            # print("Lowest cost is ", lowest_cost)
            # print("\n")

    return lowest_cost, result_city, shortest_path

def greedy_modified(G, k, weightKey = "cost"):
    lowest_cost = sys.maxsize
    result_city = -1
    shortest_path = []
    hotel_prices = nx.get_node_attributes(G, weightKey)
    for current_node in G.nodes:
        if current_node != 'A' and current_node != 'B':
            total_cost = 0
            a_path = None
            b_path = None
            if (nx.has_path(G, 'A', current_node)):
                a_path = nx.dijkstra_path(G, 'A', current_node, weight = weightKey)
            else:
                print("A has no path to city", current_node)
                continue
            if (nx.has_path(G, 'B', current_node)):
                b_path = nx.dijkstra_path(G, 'B', current_node, weight = weightKey)
            else:
                print("B has no path to city", current_node)
                continue

            a_path_cost = nx.path_weight(G, a_path, weight = weightKey)
            b_path_cost = nx.path_weight(G, b_path, weight = weightKey)
            if len(a_path) == len(b_path):
                total_cost = a_path_cost + b_path_cost
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    result_city = current_node
                    t = b_path.copy()
                    t.pop()
                    t.reverse()
                    shortest_path = a_path + t
            else:
                a_path_length = len(a_path)
                b_path_length = len(b_path)
                total_cost = a_path_cost + b_path_cost + hotel_prices[current_node] * abs(a_path_length - b_path_length)
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    result_city = current_node
                    t = b_path.copy()
                    t.pop()
                    t.reverse()
                    shortest_path = a_path + t

                k_shortest_paths_a = None
                k_shortest_paths_b = None
                if k < 1:
                    k_shortest_paths_a = list(nx.shortest_simple_paths(G, 'A', current_node, weight = weightKey))
                    k_shortest_paths_b = list(nx.shortest_simple_paths(G, 'B', current_node, weight = weightKey))
                else:
                    k_shortest_paths_a = k_shortest_paths(G, 'A', current_node, k, weight = weightKey)
                    k_shortest_paths_b = k_shortest_paths(G, 'B', current_node, k, weight = weightKey)

                shortest_path_without_hotel_a = find_shortest_path_of_length(k_shortest_paths_a, b_path_length)
                shortest_path_without_hotel_b = find_shortest_path_of_length(k_shortest_paths_b, a_path_length)

                total_cost_a = None
                total_cost_b = None
                if shortest_path_without_hotel_a is not None:
                    without_hotel_a_cost = nx.path_weight(G, shortest_path_without_hotel_a, weight = weightKey)
                    total_cost_a = without_hotel_a_cost + b_path_cost
                    if total_cost_a < lowest_cost:
                        lowest_cost = total_cost_a
                        result_city = current_node
                        t = b_path.copy()
                        t.pop()
                        t.reverse()
                        shortest_path = shortest_path_without_hotel_a + t
                if shortest_path_without_hotel_b is not None:
                    without_hotel_b_cost = nx.path_weight(G, shortest_path_without_hotel_b, weight = weightKey)
                    total_cost_b = without_hotel_b_cost + a_path_cost
                    if total_cost_b < lowest_cost:
                        lowest_cost = total_cost_b
                        result_city = current_node
                        t = shortest_path_without_hotel_b.copy()
                        t.pop()
                        t.reverse()
                        shortest_path = a_path + t

            # print("Shortest path from A to ", current_node, "is ", a_path)
            # print("Shortest path from B to ", current_node, "is ", b_path)
            # print("Current cost is ", total_cost)
            # print("Lowest cost is ", lowest_cost)
            # print("\n")

    return lowest_cost, result_city, shortest_path
    
def solve_task(G):
    print("Розв'язання задачі всіма розробленими алгоритмами:\n")
    
    solve_dijkstra(G)

    solve_greedy(G)

    solve_greedy_modified(G)

    print("Нажмите любую клавишу для продолжения...")
    msvcrt.getch()# Ожидание нажатия клавиши
    
def solve_dijkstra(G):
    total_execution_time = 0.0
    start_time = timer()
    dijkstra_shortest_path(G)
    end_time = timer()
    execution_time = (end_time - start_time)
    total_execution_time += execution_time
    print(f"Total execution time for graphs: {total_execution_time} seconds\n")
    # write_results_to_file("files/dijkstra.txt", "Алгоритм Дейкстри", )

def solve_greedy(G):
    total_execution_time = 0.0
    start_time = timer()
    lowest_cost, result_city, shortest_path = greedy(G)
    end_time = timer()
    execution_time = (end_time - start_time)
    total_execution_time += execution_time
    print("Обране місто для зустрічі: ", result_city)
    print("Фінальний шлях: ", shortest_path)
    print("Сумарні витрати на подорож (ЦФ): ", lowest_cost)
    print(f"Total execution time for Greedy Algorithm: {total_execution_time} seconds", '\n')
    write_results_to_file("files/greedy.txt", "Жадібний алгоритм", shortest_path, result_city, lowest_cost, total_execution_time)

def solve_greedy_modified(G):
    print('Greedy Algorithm Modified: ')
    print('Введіть значення k: ')
    k = int(input())
    total_execution_time = 0.0
    start_time = timer()
    lowest_cost, result_city, shortest_path = greedy_modified(G, k)
    end_time = timer()
    execution_time = (end_time - start_time)
    total_execution_time += execution_time
    print("Обране місто для зустрічі: ", result_city)
    print("Фінальний шлях: ", shortest_path)
    print("Сумарні витрати на подорож (ЦФ): ", lowest_cost)
    print(f"Total execution time for Greedy Algorithm Modified: {total_execution_time} seconds", '\n')
    write_results_to_file("files/greedy_modified.txt", "Жадібний алгоритм модифікований", shortest_path, result_city, lowest_cost, total_execution_time)

def write_results_to_file(file_name, algorithm_name, shortest_path, result_city, lowest_cost, total_execution_time):
    str0 = f"{algorithm_name}. Результати виконання.\n"
    str1 = f"Обране місто для зустрічі: {result_city}\n"
    str2 = f"Фінальний шлях: {shortest_path}\n"
    str3 = f"Сумарні витрати на подорож (ЦФ): {lowest_cost}\n"
    str4 = f"Час виконання алгоритму: {total_execution_time} секунд\n"
    file_contents = [str0, str1, str2, str3, str4]
    file = open(file_name, 'w', encoding="utf-8")
    file.writelines(file_contents)
    file.close()

def conduct_experiment():
    print("Проведення експерименту")

def print_task_solution():
    print("Виведення розв'язку задачі")
    
# Запуск меню
menu()
