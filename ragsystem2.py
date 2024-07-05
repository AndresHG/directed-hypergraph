#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ragsystem2.py
# @Author:      andres
# @Time:        7/4/24 1:06 AM


from hypergraph2 import Hypergraph
from typing import List


class RAGSystem:
    def __init__(self, hypergraph: Hypergraph):
        """Create RAGSystem object and initialize with the attached Hypergraph

        :param hypergraph: The knowledge hypergraph attached
        :type hypergraph: Hypergraph
        """
        self.hypergraph = hypergraph

    def add_knowledge(
        self, concepts: List[str], related_concepts: List[str], relation: str
    ):
        """Add knowledge to the hypergraph using the object API. We assume that
        knowledge will always come from concepts relationships.

        This works as a N-M relationship with a directed approach. This means that
        N source concepts can be related with M target concepts (related concepts).

        :param concepts: List of the source concepts
        :type concepts: List[str]
        :param related_concepts: List of the target concepts
        :type related_concepts: List[str]
        :param relation: Actual relationship between all the concepts
        :type relation: str
        """
        # TODO: validate no duplicates and so on
        # Default checks for input data
        if (
            not concepts
            or not isinstance(concepts, list)
            or len(concepts) == 0
            or not related_concepts
            or not isinstance(related_concepts, list)
            or len(related_concepts) == 0
        ):
            return
        # Loop over the source concepts to create nodes
        sources = set()
        for s_concept in concepts:
            sources.add(self.hypergraph.add_node(s_concept))

        # Loop over the related concepts to create nodes
        targets = set()
        for t_concept in related_concepts:
            targets.add(self.hypergraph.add_node(t_concept))

        # Create the hyperedge
        self.hypergraph.add_edge(sources, targets, relation)
        return

    def retrieve(self, query: str, top_k=4) -> str:
        """Search in the hypergraph for the relevant information/concepts
        and their relationships with other concepts.

        For this exercise, we don't want to repeat the same concept as an
        isolated node. Also, we are reading edges and nodes, so there may be
        chains of concepts (edges behind the scenes) or isolated concepts.

        A potential change may be to include all edges where a node is involved
        if that node has not been shown in other edge/chain.

        In order to do that I would keep a pointer of the edges bound to a node
        as part of the node information. This would help us to retrieve all the
        information related with a concept.

        :param query: Concept or information we are interested in
        :type query: str
        :param top_k: Number of results to read from index (nodes and edges). This will
            how much we want to expand or reduce the search space. Default to 4
        :type top_k: int, optional
        :return: String in Markdown format with all the information found in the hypergraph
        :rtype: str
        """
        # Find nodes in the graph based on the query
        nodes, edges = self.hypergraph.query(query, top_k=top_k)

        # Register seen nodes to not include them. Only used if isolated nodes exist
        seen_nodes = []

        # Knowledge generation from edges
        knowledge = ""
        for edge in edges:
            knowledge += "* "
            knowledge += ", ".join([node.data for node in edge.sources])
            knowledge += f" - {edge.relation} - "
            knowledge += ", ".join([node.data for node in edge.targets])
            knowledge += "\n"
            # Update seen nodes
            seen_nodes.extend([node.id for node in edge.sources])
            seen_nodes.extend([node.id for node in edge.targets])

        # Check isolated nodes. Depending on the problem definition, this can
        # be excluded. If it is mandatory to provide a relationship when adding
        # knowledge, then we don't need to search for isolated notes.
        for node in nodes:
            if node.id in seen_nodes:
                continue
            knowledge += f"* {node.data}"
            knowledge += "\n"

        return knowledge
