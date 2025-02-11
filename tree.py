from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, Self, Optional
from deap import gp
import pygraphviz as pgv, os, numpy as np
from matplotlib import pyplot as plt


TreeNode = gp.Primitive | gp.Terminal

def show_img(img, title, save_to=None):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    if save_to is None:
        plt.title(title)
        plt.show()
    else:
        plt.savefig(save_to, bbox_inches='tight')
        plt.close()

# Define a type variable
T = TypeVar('T')
@dataclass
class Box(Generic[T]):
    value: T

@dataclass
class Tree[T]:
    function: TreeNode
    children: list["Tree[Any]"]
    pset: gp.PrimitiveSetTyped
    value: Optional[T] = None


    def __repr__(self) -> str:
        return self.function.format(*self.children)

    def compile(self) -> Callable[..., T]:
        return gp.compile(self, pset=self.pset)

    def id(self) -> str:
        return str(id(self))

    def _evaluate_all_nodes(self, *args) -> Any:
        self.value = self.compile()(*args)

        if is_image(self.value):
            os.makedirs("_treedata", exist_ok=True)
            show_img(self.value, '', save_to=f'_treedata/{self.id()}.png')


        for child in self.children:
            child._evaluate_all_nodes(*args)

    @staticmethod
    def construct_tree(model: list[TreeNode], pset: gp.PrimitiveSetTyped, index: Box[int]) -> "Tree":
        function = model[index.value]
        index.value += 1
        return Tree(function, [Tree.construct_tree(model, pset, index) for _ in range(function.arity)], pset)

    @staticmethod
    def of(model: list[TreeNode], pset: gp.PrimitiveSetTyped) -> "Tree":
        return Tree.construct_tree(model, pset, Box(0))

    def nodes(self) -> list["Tree"]:
        return [self] + sum((child.nodes() for child in self.children), [])


@dataclass
class TreeDrawer:
    drawer: list[tuple[Callable[[Tree], bool], Callable[[pgv.AGraph, Tree], None]]] = field(default_factory=list)

    def __post_init__(self):
        self.drawer = []
        self.register_draw_function(lambda t: is_image(t.value), draw_image)\
        .register_draw_function(lambda t: t.function.arity == 0 and "ARG" not in t.function.name, lambda *_: None)

    def register_draw_function(self, predicate: Callable[[Tree], bool], draw_function: Callable) -> Self:
        self.drawer.append((predicate, draw_function))
        return self

    def get_graph(self, tree: Tree, *args: Any) -> pgv.AGraph:
        tree._evaluate_all_nodes(*args)

        graph = pgv.AGraph(strict=False, directed=True)

        # requires a directory to store images in :(
        if not os.path.isdir('_treedata'):
            os.makedirs('_treedata')

        self._populate_graph(tree, graph)
        graph.layout(prog="dot")
        return graph

    def save_graph(self, file: str, *args) -> None:
        self.get_graph(*args).draw(file)


    def _populate_graph(self, tree: Tree, graph: pgv.AGraph) -> None:
        if tree.function.arity == 0:
            graph.add_node(tree.id(), label=tree.function.format())
        else:
            graph.add_node(tree.id(), label=tree.function.name)

        self._displayvalue(tree, graph)

        for child in tree.children:
            self._populate_graph(child, graph)
            graph.add_edge(tree.id(), child.id(), dir="back")

    def _displayvalue(self, tree: Tree, graph: pgv.AGraph) -> None:
        for predicate, draw_function in self.drawer:
            if predicate(tree):
                draw_function(graph, tree)
                break
        else:
            graph.add_node(f"{tree.id()}result", label=f"{tree.value}", shape="plaintext")

        if not graph.has_node(f"{tree.id()}result"):
            return None

        graph.add_edge(tree.id(), f"{tree.id()}result", style="invis", dir="both")

        result_holder = graph.add_subgraph([tree.id(),f"{tree.id()}result"], name=f"{tree.id()}-resultholder")
        result_holder.graph_attr['rank']='same'


def is_image(value: Any) -> bool:
    return isinstance(value, np.ndarray) and len(value.shape) == 2

def draw_image(graph: pgv.AGraph, tree: Tree):
    graph.add_node(f"{tree.id()}result", image=f'_treedata/{tree.id()}.png', label="", imagescale=True, fixedsize=True, shape="plaintext", width=2, height=2)
