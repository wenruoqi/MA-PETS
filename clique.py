def is_clique(graph, vertices):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not graph[vertices[i]][vertices[j]]:
                return False
    return True

def min_covering_clique(graph, vertices, k, n, curr_vertices=[]):
    if len(curr_vertices) == k:
        return is_clique(graph, curr_vertices)

    if n == len(vertices):
        return False

    # Case 1: Include the current vertex
    curr_vertices.append(vertices[n])
    if min_covering_clique(graph, vertices, k, n + 1, curr_vertices):
        return True

    # Case 2: Exclude the current vertex
    curr_vertices.pop()
    return min_covering_clique(graph, vertices, k, n + 1, curr_vertices)

def find_min_covering_clique(graph):
    n = len(graph)
    vertices = list(range(n))

    for k in range(1, n + 1):
        if min_covering_clique(graph, vertices, k, 0, []):
            return k

    return -1

# Example Usage
# graph represented as an adjacency matrix

graph = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]

print(find_min_covering_clique(graph))
