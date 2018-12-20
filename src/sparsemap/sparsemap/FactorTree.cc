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

// Backtrack to obtain the labeled spans
// (ParseNodes) that make up the Viterbi tree.
void Backtrack(vector<vector<vector<vector<int> > > > path,  // TODO: make this a pointer?
               vector<AD3::ParseNode> *nodes,
               int label,
               int left,
               int right,
               int length) {
  // Add the node to the list of nodes.
  nodes->push_back(ParseNode(label, left, right));  // TODO: is this really all correct? Pointers and all??
  // If the node spans more than one word we recursively
  // add the children.
  if (right > left + 1) {
    int split = path[label][0][length][0];
    int left_label = path[label][0][length][1];
    int right_label = path[label][0][length][2];
    Backtrack(path, nodes, left_label, left, split, length);
    Backtrack(path, nodes, right_label, split, right, length);
  }
}

// The score for an edge incoming to node (label, left, right).
// Note that all edges incoming to a node get the same score.
double FactorTree::GetEdgeScore(int label,
                    int left,
                    int right,
                    const vector<double> &variable_log_potentials) {
  std::cout << "FactorTree.cc >> GetEdgeScore >>  " << label << " " << left << " " << right << std::endl;
  std::cout << "FactorTree.cc >> GetEdgeScore >> index_nodes_ size: " << index_nodes_.size();

  int index = index_nodes_[label][left][right];  // FIXME: throws segmentation fault...
  std::cout << "FactorTree.cc >> GetEdgeScore >> index: " << index << std::endl;

  return variable_log_potentials[index];
}

// Decoder for for the forest using the Viterbi algorithm.
int FactorTree::RunViterbi(const vector<double>& variable_log_potentials,
                           vector<ParseNode> *nodes,
                           double *value) {

  std::cout << "FactorTree.cc >> RunViterbi: " << std::endl;

  // Vector of shape [num_labels, length, length] initialized with -1.
  vector<vector<vector<double> > >  chart(
    num_labels_, vector<vector<double> >(length_ + 1, vector<double>(length_ + 1, -1)));
  // Vector of shape [num_labels, length, length, 3] initialized with -1.
  vector<vector<vector<vector<int> > > > path(
    num_labels_, vector<vector<vector<int> > >(length_ + 1, vector<vector<int> >(length_ + 1, vector<int>(3, -1))));
  // Source: https://stackoverflow.com/questions/29305621/problems-using-3-dimensional-vector

  std::cout << "FactorTree.cc >> RunViterbi >> chart size: " << chart.size() << chart[0].size() << chart[0][0].size() << std::endl;
  std::cout << "FactorTree.cc >> RunViterbi >> path size: " << path.size() << path[0].size() << path[0][0].size() << path[0][0][0].size() << std::endl;

  int root = 0;
  double best_value;

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
            double val = edge_score + best_left_value + best_right_value;
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


  std::cout << "FactorTree.cc >> RunViterbi >> best_value: " << best_value << std::endl;

  // Path (node sequence) backtracking.
  // vector<int> *sequence = static_cast<vector<int>*>(configuration);
  // assert(sequence->size() == length);
  // (*sequence)[length - 1] = best;
  // for (int i = length - 1; i > 0; --i) {
  //   (*sequence)[i - 1] = path[i][(*sequence)[i]];
  // }

  // Path backracking puts the recognized nodes in `nodes`.
  Backtrack(path, nodes, root, 0, length_, length_);

  // Store the value of the viterbi tree.
  *value = best_value;

  } // RunViterbi

} // namespace AD3

#endif // FACTOR_TREE_H_
