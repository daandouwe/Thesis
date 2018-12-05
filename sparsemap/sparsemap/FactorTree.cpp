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

// The score for all the edges that are incoming to edge (label, left, right)
double GetEdgeScore(int label,
                    int left,
                    int right,
                    const vector<double> &variable_log_potentials) {
  int index = index_edges_[label][left][right];
  return variable_log_potentials[index];
}


// Decoder for for the forest using the Viterbi algorithm.
int FactorTree::RunViterbi(const vector<double>& variable_log_potentials,
                           vector<Node> *nodes,
                           double *value) {
  // Decode using the Viterbi algorithm.

  int j;
  double edge_score;
  double left_val;
  double right_val;
  // TODO: find out how to asign size here.
  vector<vector<vector<double> > > > chart;  // should be [num_labels, length, length]
  vector<vector<vector<vector<int> > > > > path;  // should be [num_labels, length, length, 3]

  // Visit nodes of the complete forest in topological order.
  for (int s = 1; s < length_ + 1; ++s) {
    for (int i = 0; i < length_ + 1 - s; i++) {
      j = i + s;
      for (int l = 0; l < num_labels_; l++) {
        // Current node is (l, i, j).
        edge_score = GetEdgeScore(l, i, j, variable_log_potentials);

        if (j == i + 1) {
          chart[l][i][j] = edge_score;
          path[l][i][j][0] = -1;
          path[l][i][j][1] = -1;
          path[l][i][j][2] = -1;
        } else {
          // Get the best incoming edge (k, B, C),
          // where k is the split-point and B, C
          // the left and right child label respectively.
          double best_value;
          int best = -1;
          int best_split;
          int best_left_label;
          int best_right_label;

          // Get the best point.
          for (int k = 0; k < num_labels_; k++) {

            // Get the best label for the left span
            // given the current split-point.
            double best_left_value;
            double best_left = -1;
            for (int b = 0; b < num_labels_; b++) {
              left_val = chart[b][i][k];
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
              right_val = chart[c][k][j];
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
  // that make up the Viterbi tree.

  void Backtrack(int label, int left, int right, int index) {
    // TODO: This looks really sketchy... and is probably broken.
    (*nodes)[i] = Node(label, left, right);
    ++index;
    if (right > left + 1) {
      split = chart[root][0][length_][0];
      left_label = chart[root][0][length_][1];
      right_label = chart[root][0][length_][2];
      Backtrack(left_label, left, split, index);
      Backtrack(right_label, split, right, index);
    }
  }

  int index = 0;
  Backtrack(root, 0, length_, index);

  // Path (node sequence) backtracking.
  // vector<int> *sequence = static_cast<vector<int>*>(configuration);
  // assert(sequence->size() == length);
  // (*sequence)[length - 1] = best;
  // for (int i = length - 1; i > 0; --i) {
  //   (*sequence)[i - 1] = path[i][(*sequence)[i]];
  // }

  *value = best_value;
} // RunViterbi

} // namespace AD3

#endif // FACTOR_TREE_H_
