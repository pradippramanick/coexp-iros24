from reflect.constants import NAME_MAP


class Node(object):
    def __init__(self, name):
        if name in NAME_MAP:
            self.name = NAME_MAP[name]
        else:
            self.name = name.lower()
        self.name_w_state = None

    def set_state(self, state):
        self.name_w_state = state

    def __str__(self):
        return self.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return True if self.get_name() == other.get_name() else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name

    def get_name_w_state(self):  # only rerun nodes that has a state
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return None


class Edge(object):
    def __init__(self, start_node, end_node, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type

    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + "->" + self.edge_type + "->" + str(self.end)


class mSceneGraph(object):
    """
    Create a spatial scene graph
    """

    def __init__(self, event, task):
        self.nodes = []
        self.total_nodes = []
        self.edges = {}

    def add_node_wo_edge(self, node):
        self.total_nodes.append(node)

    def in_robot_gripper(self):
        for obj in self.event.metadata["objects"]:
            if obj["isPickedUp"]:
                return obj['objectId']
        return None

    def __eq__(self, other):
        if (set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values())):
            return True
        else:
            return False

    def __str__(self):
        visited = []
        res = "[Nodes]:\n"
        for node in set(self.nodes):
            res += node.get_name()
            res += "\n"
        res += "\n"
        res += "[Edges]:\n"
        for edge_key, edge in self.edges.items():
            name_1, name_2 = edge_key
            edge_key_reversed = (name_2, name_1)
            if (edge_key not in visited and edge_key_reversed not in visited) or edge.edge_type in ['on top of',
                                                                                                    'inside',
                                                                                                    'occluding']:
                res += str(edge)
                res += "\n"
            visited.append(edge_key)
        return res
