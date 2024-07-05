#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main.py
# @Author:      AndresHG
# @Time:        7/1/24 10:33 PM


from hypergraph2 import Hypergraph
from ragsystem2 import RAGSystem


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Example usage
    hypergraph = Hypergraph()
    rag_system = RAGSystem(hypergraph)
    # Populate the hypergraph with programming knowledge
    rag_system.add_knowledge(
        ["Python"], ["OOP", "Interpreted", "DynamicTyping"], "language_features"
    )
    rag_system.add_knowledge(
        ["List Comprehension"],
        ["Python", "Functional Programming"],
        "programming_technique",
    )
    rag_system.add_knowledge(
        ["Django", "Pyramid"], ["Python", "Web Framework", "ORM"], "framework"
    )
    rag_system.add_knowledge(["Java", "C"], ["Compiled", "Typed"], "language_features")
    # Retrieve information
    result = rag_system.retrieve("Web framework")
    print(result)
