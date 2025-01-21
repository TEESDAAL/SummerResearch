from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
from deap import gp
import pygraphviz as pgv, os, numpy as np
from matplotlib import pyplot as plt

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
class Tree:
    function: Any
    children: list["Tree"]
    pset: gp.PrimitiveSetTyped
    _result: Any = None
    _node_num: int = None

    def __iter__(self):
        return iter((self.function, self.children))

    def __repr__(self):
        return self.function.format(*self.children)

    def compile(self) -> Callable:
        return gp.compile(self, pset=self.pset)

    def get_graph(self, *args) -> pgv.AGraph:
        self._evaluate_all_nodes(*args)

        graph = pgv.AGraph(strict=False, directed=True)
        if not os.path.isdir('_treedata'):
            os.makedirs('_treedata')
        self._populate_graph(graph)
        graph.layout(prog="dot")
        return graph

    def save_graph(self, file: str, *args) -> None:
        self.get_graph(*args).draw(file)


    def _populate_graph(self, graph: pgv.AGraph) -> None:

        if self.function.arity == 0:
            graph.add_node(self.id(), label=self.function.format())
        else:
            graph.add_node(self.id(), label=self.function.name)

        self._display_result(graph)

        for child in self.children:
            child._populate_graph(graph)
            graph.add_edge(self.id(), child.id(), dir="back")

    def id(self) -> str:
        return str(id(self))

    def _display_result(self, graph):
        if self.function.arity == 0 and "ARG" not in self.function.name:
            return
        if isinstance(self._result, np.ndarray):
            graph.add_node(f"{self.id()}result", image=f'_treedata/{self.id()}.png', label="", imagescale=True, fixedsize=True, shape="plaintext", width=2, height=2)
        else:
            graph.add_node(f"{self.id()}result", label=f"{self._result}", shape="plaintext")

        graph.add_edge(self.id(), f"{self.id()}result", style="invis", dir="both")
        B=graph.add_subgraph([self.id(),f"{self.id()}result"],name=f"{self.id()}-resultholder")
        B.graph_attr['rank']='same'

    def _evaluate_all_nodes(self, *args) -> Any:
        self._result = self.compile()(*args)

        if isinstance(self._result, np.ndarray):
            os.makedirs("_treedata", exist_ok=True)
            show_img(self._result, '', save_to=f'_treedata/{self.id()}.png')


        for child in self.children:
            child._evaluate_all_nodes(*args)

    @staticmethod
    def construct_tree(model: list[gp.PrimitiveTree], pset: gp.PrimitiveSetTyped, index: Box[int]) -> "Tree":
        function = model[index.value]
        index.value += 1
        return Tree(function, [Tree.construct_tree(model, pset, index) for _ in range(function.arity)], pset)

    @staticmethod
    def of(model: list[gp.PrimitiveTree], pset: gp.PrimitiveSetTyped) -> "Tree":
        return Tree.construct_tree(model, pset, Box(0))

    def nodes(self) -> list["Tree"]:
        return [self] + sum((child.nodes() for child in self.children), [])
