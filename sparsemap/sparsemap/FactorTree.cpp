// Copyright (c) 2018 Daan van Stigt
// All Rights Reserved.

#ifndef FACTOR_TREE_H_
#define FACTOR_TREE_H_

#include "FactorTree.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iterator>
#include <cstdlib>
#include <limits>
#include <cerrno>

using namespace std;

namespace AD3 {

// The score for an edge incoming to node (label, left, right).
// Note that all edges incoming to a node get the same score.
double GetEdgeScore(int label,
                    int left,
                    int right,
                    const vector<double> &variable_log_potentials) {
  int index = index_nodes_[label][left][right];
  return variable_log_potentials[index];
}


// Decoder for for the forest using the Viterbi algorithm.
int FactorTree::RunViterbi(const vector<double>& variable_log_potentials,
                           vector<Node> *nodes,
                           double *value) {

  // Vector of shape [num_labels, length, length] initialized with -1.
  vector<vector<vector<double> > >  chart(
    num_labels_, vector<vector<double> >(length_, vector<double>(length_, -1))));
  // Vector of shape [num_labels, length, length, 3] initialized with -1.
  vector<vector<vector<vector<int> > > > path(
    num_labels_, vector<vector<vector<int> > >(length_, vector<vector<int> >(length_, vector<int>(3, -1)))));
  // Source: https://stackoverflow.com/questions/29305621/problems-using-3-dimensional-vector

  // Visit nodes of the complete forest in topological order.
  for (int s = 1; s < length_ + 1; ++s) {
    for (int i = 0; i < length_ + 1 - s; i++) {
      int j = i + s;  // span of length s from i to j
      for (int l = 0; l < num_labels_; l++) {
        // Current node is (l, i, j).
        // Each edge coming in to this node has the same score.
        double edge_score = GetEdgeScore(l, i, j, variable_log_potentials);

        if (j == i + 1) {
          chart[l][i][j] = edge_score;
        } else {
          // Get the best incoming edge (k, B, C),
          // where k is the split-point and B, C
          // the left and right child label respectively.
          double best_value;
          int best = -1;
          int best_split;
          int best_left_label;
          int best_right_label;

          // Get the best split point.
          for (int k = i + 1; k < j; k++) {

            // Get the best label for the left span
            // given the current split-point.
            double best_left_value;
            double best_left = -1;
            for (int b = 0; b < num_labels_; b++) {
              double left_val = chart[b][i][k];
              if (best_left < 0 || left_val > best_left_value) {
                best_left_value = left_val;
                best_left = b;
              }
            }

            // Get the best label for the right span
            // given the current split-point.
            double best_right_value;
            double best_right = -1;
            for (int c = 0; c < num_labels_; c++) {
              double right_val = chart[c][k][j];
              if (best_right < 0 || right_val > best_right_value) {
                best_right_value = right_val;
                best_right = c;
              }
            }

            // Get the total score of this edge.
            double val = edge_score + best_left_val + best_right_val;
            // Store the current incoming node if it is the best.
            if (best < 0 || val > best_value) {
              best_value = val;
              best = k;
              best_split = k;
              best_left_label = best_left;
              best_right_label = best_right;
            }
          }
          // Save the best incoming edge.
          chart[l][i][j] = best_value;
          path[l][i][j][0] = best_split;
          path[l][i][j][1] = best_left_label;
          path[l][i][j][2] = best_right_label;
        }
      }
    }
  }

  // Backtrack to obtain the labeled spans
  // (Nodes) that make up the Viterbi tree.
  void Backtrack(vector<vector<vector<vector<int> > > > path,  // TODO: make this a pointer?
                 vector<Node> *nodes,
                 int label,
                 int left,
                 int right) {
    // Add the node to the list of nodes.
    nodes->push_back(Node(label, left, right));  // TODO: is this really all correct? Pointers and all??
    // If the node spans more than one word we recursively
    // add the children.
    if (right > left + 1) {
      split = path[root][0][length_][0];
      left_label = path[root][0][length_][1];
      right_label = path[root][0][length_][2];
      Backtrack(path, nodes, left_label, left, split);
      Backtrack(path, nodes, right_label, split, right);
    }
    // TODO: resize nodes to be of the right size
  }

  // Path backrackin puts the recognized nodes in `nodes`.
  Backtrack(path, nodes, root, 0, length_, index);

  // Store the value of the viterbi tree.
  *value = best_value;

  // Path (node sequence) backtracking.
  // vector<int> *sequence = static_cast<vector<int>*>(configuration);
  // assert(sequence->size() == length);
  // (*sequence)[length - 1] = best;
  // for (int i = length - 1; i > 0; --i) {
  //   (*sequence)[i - 1] = path[i][(*sequence)[i]];
  // }

} // RunViterbi

} // namespace AD3

#endif // FACTOR_TREE_H_
