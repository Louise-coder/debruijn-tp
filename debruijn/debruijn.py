#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import itertools
import random
from random import randint
import statistics
import textwrap
from typing import Iterator, Dict, List
import argparse
import os
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
)


random.seed(9001)

matplotlib.use("Agg")

__author__ = "Louise LAM"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Louise LAM"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Louise LAM"
__email__ = "louise.lam@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"{sys.argv[0]} -h"
    )
    parser.add_argument(
        "-i",
        dest="fastq_file",
        type=isfile,
        required=True,
        help="Fastq file",
    )
    parser.add_argument(
        "-k",
        dest="kmer_size",
        type=int,
        default=22,
        help="k-mer size (default 22)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f",
        dest="graphimg_file",
        type=Path,
        help="Save graph as an image (png)",
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r", encoding="utf-8") as file:
        while True:
            try:
                fastq_iter = iter(file)  # itérateur sur le fichier
                next(fastq_iter)  # header
                sequence = next(fastq_iter).strip()
                next(fastq_iter)  # plus
                next(fastq_iter)  # quality
                yield sequence
            except StopIteration:
                break


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(0, len(read) - kmer_size + 1):
        kmer = read[i : i + kmer_size]
        yield kmer


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file):
        kmers = cut_kmer(read, kmer_size)
        for kmer in kmers:
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    for kmer, count in kmer_dict.items():
        graph.add_edge(kmer[:-1], kmer[1:], weight=count)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        path_size = len(path)
        for i in range(path_size - 1):
            node_u = path[i]
            node_v = path[i + 1]
            if graph.has_edge(node_u, node_v):
                graph.remove_edge(node_u, node_v)
        nodes_to_delete = []
        if delete_entry_node and delete_sink_node:
            nodes_to_delete = path
        elif delete_entry_node:
            nodes_to_delete = path[:-1]
        elif delete_sink_node:
            nodes_to_delete = path[1:]
        else:
            nodes_to_delete = path[1:-1]
        for node in nodes_to_delete:
            graph.remove_node(node)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if statistics.stdev(weight_avg_list) > 0:
        index_best_path = weight_avg_list.index(max(weight_avg_list))
    elif statistics.stdev(path_length) > 0:
        index_best_path = path_length.index(max(path_length))
    else:
        index_best_path = randint(0, len(path_list) - 1)
    del path_list[index_best_path]
    graph = remove_paths(
        graph, path_list, delete_entry_node, delete_sink_node
    )
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [
            d["weight"]
            for (u, v, d) in graph.subgraph(path).edges(data=True)
        ]
    )


def solve_bubble(
    graph: DiGraph, ancestor_node: str, descendant_node: str
) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = list(
        all_simple_paths(graph, ancestor_node, descendant_node)
    )
    path_length = [len(path) for path in path_list]
    weight_avg_list = [
        path_average_weight(graph, path) for path in path_list
    ]
    graph = select_best_path(
        graph, path_list, path_length, weight_avg_list, False, False
    )
    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    is_bubble = False
    node_n = None
    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for i, j in itertools.combinations(predecessors, 2):
                ancestor_node = lowest_common_ancestor(graph, i, j)
                if ancestor_node is not None:
                    is_bubble = True
                    node_n = node
                    break
        if is_bubble:
            node_n = node
            break
    if is_bubble:
        graph = simplify_bubbles(
            solve_bubble(graph, ancestor_node, node_n)
        )
    return graph


def solve_tips(
    graph: DiGraph,
    common_node: str,
    tips: List[str],
    tip_type: str,
) -> DiGraph:
    """Explore and solve tip issue.

    :param graph: (nx.DiGraph) A directed graph object.
    :param common_node: (str) The common node.
    :param tips: (List[str]) A list of tip nodes to resolve.
    :param tip_type: (str) Either "entry" or "ending".
    """
    path_list = []
    if tip_type == "entry":
        for tip in tips:
            path_list += list(all_simple_paths(graph, tip, common_node))
        delete_entry_node, delete_sink_node = True, False
    else:
        for tip in tips:
            path_list += list(all_simple_paths(graph, common_node, tip))
        delete_entry_node, delete_sink_node = False, True
    path_length = [len(path) for path in path_list]
    weight_avg_list = [
        path_average_weight(graph, path) for path in path_list
    ]
    graph = select_best_path(
        graph,
        path_list,
        path_length,
        weight_avg_list,
        delete_entry_node,
        delete_sink_node,
    )
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    is_tip = False
    entries = []
    common_node = None
    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for s_node in starting_nodes:
                if has_path(graph, s_node, node):
                    entries.append(s_node)
            if len(entries) > 1:
                is_tip = True
                common_node = node
                break
    if is_tip:
        new_graph = solve_tips(graph, common_node, entries, "entry")
        graph = solve_entry_tips(
            new_graph,
            get_starting_nodes(new_graph),
        )
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    is_tip = False
    endings = []
    common_node = None
    for node in graph.nodes:
        successors = list(graph.successors(node))
        if len(successors) > 1:
            for e_node in ending_nodes:
                if has_path(graph, node, e_node):
                    endings.append(e_node)
            if len(endings) > 1:
                is_tip = True
                common_node = node
                break
    if is_tip:
        new_graph = solve_tips(graph, common_node, endings, "ending")
        graph = solve_entry_tips(
            new_graph,
            get_sink_nodes(new_graph),
        )
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes():
        predecessors = graph.predecessors(node)
        if not list(predecessors):
            starting_nodes.append(node)
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node in graph.nodes():
        successors = graph.successors(node)
        if not list(successors):
            sink_nodes.append(node)
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for s_node in starting_nodes:
        for e_node in ending_nodes:
            all_paths = all_simple_paths(graph, s_node, e_node)
            for path in list(all_paths):
                contig = path[0] + "".join(node[-1] for node in path[1:])
                contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for i, (contig, contig_size) in enumerate(contigs_list):
            file.write(f">contig_{i} len={contig_size}\n")
            file.write(f"{textwrap.fill(contig, width=80)}\n")


def draw_graph(
    graph: DiGraph, graphimg_file: Path
) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    _, _ = plt.subplots()
    elarge = [
        (u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3
    ]
    # print(elarge)
    esmall = [
        (u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3
    ]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=esmall,
        width=6,
        alpha=0.5,
        edge_color="b",
        style="dashed",
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    # Lecture du fichier et construction du graphe
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)
    # Résolution des bulles
    graph = simplify_bubbles(graph)
    # Résolution des pointes d’entrée et de sortie
    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))
    # Ecriture du/des contigs
    contigs = get_contigs(
        graph, get_starting_nodes(graph), get_sink_nodes(graph)
    )
    save_contigs(contigs, args.output_file)
    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
