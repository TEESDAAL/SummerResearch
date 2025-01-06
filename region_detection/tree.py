from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Sequence, Union
from deap import gp
import pygraphviz as pgv, os, numpy as np
from matplotlib import pyplot as plt, patches

def show_img(img, title, regions=None, save_to=None):

    fig, ax = plt.subplots()

    image = ax.imshow(img, cmap="gray")
    plt.colorbar(image, ax=ax)
    if regions is not None:
        for region in regions:

            rect = patches.Rectangle((region[0], region[1]), region[2], region[3], linewidth=3, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
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
Predicate = Callable[[Any], bool]

@dataclass
class Tree:
    function: Any
    children: list["Tree"]
    pset: gp.PrimitiveSetTyped
    _input: Any = None
    _result: Any = None

    def display_methods(self) -> Sequence[tuple[Predicate, Callable[[pgv.AGraph, Any, Any], None]]]:
        return [
            (lambda data: isinstance(data, np.ndarray),
             lambda graph, img, _: self.add_image(graph, img)),
            (lambda data: isinstance(data, list),
            lambda graph, regions, inputs: self.add_text(graph, self.add_image(graph, inputs[0], hightligted_regions=regions)))
        ] + [(lambda _: True, lambda graph, data, _: self.add_text(graph, data))]

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
        for predicate, drawing_function in self.display_methods():
            if predicate(self._result):
                drawing_function(graph, self._result, self._input)
                break

        graph.add_edge(self.id(), f"{self.id()}result", style="invis", dir="both")
        B=graph.add_subgraph([self.id(),f"{self.id()}result"],name=f"{self.id()}-resultholder")
        B.graph_attr['rank']='same'

    def add_text(self, graph, data) -> None:
        graph.add_node(f"{self.id()}result", label=f"{data}", shape="plaintext")

    def add_image(self, graph, data, hightligted_regions=None, width: float=2, height: float=2) -> None:
        path = f'_treedata/{self.id()}.png'
        os.makedirs("_treedata", exist_ok=True)
        show_img(data, '', save_to=path, regions=hightligted_regions)

        graph.add_node(f"{self.id()}result", image=f'_treedata/{self.id()}.png', label="", imagescale=True, fixedsize=True, shape="plaintext", width=width, height=height)
        #os.remove(path)

    def _evaluate_all_nodes(self, *args) -> None:
        self._input = args
        self._result = self.compile()(*args)

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
