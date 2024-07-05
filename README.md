# Directed Hypergraph

This is small project to visualize how to create (based on my own 
knowledge) a hypergraph. There may be multiple ways to implement this, but 
this is just the approach I have chosen.

Main idea is to create the Class structure, define the main functions and
propose  techniques or improvements over a baseline.

There is also a small `requirements.txt` file to show the required 
packages (very few for this framework), but on a proper scenario I would
use something like Poetry or Pipenv (lock files are sometime a mess, but
they work!).

## Main concepts

* Search Optimization:
* Parameter checking:
* Storage: For storage I propose to use the incidence matrix H where we 
can represent for each pair of node-edge how they are related. 
  * Matrix values would be: 1 if the node is a source of the edge, 0 if 
  the node is not in the edge and -1 if the node is a target of the edge.
  * Index from Facebook FAISS can be dumped directly