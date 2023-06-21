import networkx as nx
import matplotlib.pyplot as plt

def fill_graph():
    G = nx.Graph()

    # Ввод количества вершин в графе
    num_vertices = int(input("Введите количество вершин в графе: "))

    # Ввод вершин и их весов
    for i in range(num_vertices):
        i += 1
        vertex = str(i)
        weight = int(input(f"Введите вес для вершины {vertex}: "))
        G.add_node(vertex, weight=weight)

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
            edge_weight = (input(f"Введите вес для ребра ({u}, {v}): "))
            G.add_edge(u, v, weight=edge_weight)
        else:
            print("Ошибка: одна или обе вершины не существуют в графе.")

    return G

    

# Пример использования
graph = fill_graph()


def print_graph(G):
    # Вывод графа
    print("Граф:")

    # Рисуем граф
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='green', node_size=800, edge_color='gray', width=1, alpha=0.7)

    # Выводим веса вершин
    labels = nx.get_node_attributes(G, 'weight')
    node_labels = {node: f"\n\n{weight}" for node, weight in labels.items()}

    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=800, edge_color='gray', width=1, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, verticalalignment='top')



    # Выводим веса ребер
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Показываем граф на экране
    plt.show()


print_graph(graph)
