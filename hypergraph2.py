#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    hypergraph2.py
# @Author:      andres
# @Time:        7/2/24 6:46 PM

import faiss
from sentence_transformers import SentenceTransformer

import uuid
from typing import Dict, Set, Tuple


class Node:
    def __init__(self, data: str):
        """Create a node.

        Reference in retrival function of the RAGSystem: A potential change
        may be to include all edges where a node is involved.

        In order to do that I would keep a pointer of the edges bound to a node
        as part of the node information. This would help us to retrieve all the
        information related with a concept.

        :param data: The information stored in the node
        :type data: str
        """
        self.id = uuid.uuid4()
        self.data = data


class Hyperedge:
    def __init__(self, sources: Set[Node], targets: Set[Node], relation: str):
        """Create a directed hyperedge.

        :param sources: Source nodes for the hyperedge
        :type sources: Set[Node]
        :param targets: Target nodes for the hyperedge
        :type targets: Set[Node]
        :param relation: String representing the relationship between the nodes
        :type relation: str
        """
        self.id = uuid.uuid4()
        self.sources = sources
        self.targets = targets
        self.relation = relation


class Hypergraph:
    def __init__(self):
        """
        Initialize hypergraph (nodes and edges) and the index for search.

        For `self._index` we are using `IndexFlatL2` from `faiss` library. This
        will vector index for our nodes descriptions, so we can find information
        easily.

        For `self.transformer` (object that will encode our information), bert-like
        embeddings approach has been selected. From `sentence_transformer` library,
        currently using `bert-base-nli-mean-tokens`.
        """
        self.nodes: Dict[uuid.UUID, Node] = {}
        self.edges: Dict[uuid.UUID, Hyperedge] = {}

        self.transformer = SentenceTransformer("bert-base-nli-mean-tokens")
        sentence_embeddings = self.transformer.encode(["Dummy text", "Dummy 2"])
        d = sentence_embeddings.shape[1]

        self._index = faiss.IndexFlatL2(d)
        self._index_objs = []

    def add_node(self, data: str) -> Node:
        """Add the node to the hypergraph. If it already exists, return the
        existing node.

        :param data: Information for the given node
        :type data: str
        :return: Pointer to the new/existing instance of the node
        :rtype: Node
        """
        D, I = self._index.search(self.transformer.encode([data]), 1)
        # TODO: improve this. It is an arbitrary number that rounded will be 0
        if D[0][0] > 0.0001:  # Node exist already
            node = Node(data)
            self.nodes[node.id] = node

            self._index.add(self.transformer.encode([data]))
            self._index_objs.append(node)
        else:
            # Get the first and only item of the index
            node = self._index_objs[I[0][0]]
        return node

    def add_edge(
        self, sources: Set[Node], targets: Set[Node], relation: str
    ) -> Hyperedge:
        """Add edges to the hypergraph given the nodes. Since this is directed graph,
        we have source and target nodes.

        :param sources: Set of nodes to include as sources
        :type sources: Set[Node]
        :param targets:  Set of target nodes
        :type targets: Set[Node]
        :param relation: str, identify the actual relation between the nodes
        :type relation: str
        :return: Pointer to the object created
        :rtype: Hyperedge
        """
        edge = Hyperedge(sources, targets, relation)
        self.edges[edge.id] = edge

        # We are adding one index per target. Reason why is: OOP is related to Python
        # but OOP is not related to interpreted. So targets should be separated
        index_term = (
            ", ".join([node.data for node in sources])
            + " - "
            + relation
            + " - "
            + ", ".join([node.data for node in targets])
        )

        # Update index amd index_objs
        self._index.add(self.transformer.encode([index_term]))
        self._index_objs.append(edge)
        return edge

    def query(self, criteria: str, top_k: int = 4) -> Tuple[Set[Node], Set[Hyperedge]]:
        """Find for nodes and edges matching the requested criteria. We are using
        `IndexFlatL2` based on vector Euclidean distance to find similar information.

        :param criteria: The information we are looking for
        :type criteria: str
        :param top_k: Number of results to retrieve from the index. Default 4
        :type top_k: int, optional
        :return:
        """
        # Search similar nodes and edges based on the criteria
        encoded_criteria = self.transformer.encode([criteria])
        D, I = self._index.search(encoded_criteria, top_k)
        results = [self._index_objs[i] for i in I[0]]

        ret_nodes = set()
        ret_edges = set()
        # Iterate over nodes and select the returning ones
        for i, result in enumerate(results):
            # Read node or edge nodes and append to the returning list
            if isinstance(result, Node):
                ret_nodes.add(result)
            if isinstance(result, Hyperedge):
                ret_edges.add(result)

        return ret_nodes, ret_edges
