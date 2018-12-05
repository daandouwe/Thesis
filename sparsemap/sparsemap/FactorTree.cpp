// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.1.
//
// AD3 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.1.  If not, see <http://www.gnu.org/licenses/>.

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


// Decoder for the basic model; it finds a maximum weighted arborescence
// using Chu-Liu-Edmonds' algorithm.
int FactorTree::RunViterbi(const vector<double>& variable_log_potentials,
                           vector<Node> *nodes,
                           double *value) {
  // Decode using the Viterbi algorithm.

  int j;
  double edge_score;
  double left_val;
  double right_val;
  vector<vector<vector<double> > > > chart;
  vector<vector<vector<int> > > > path;

  // Visit nodes of the complete forest in topological order.
  for (int s = 1; s < length_ + 1; ++s) {
    for (int i = 0; i < length_ + 1 - s; i++) {
      j = i + s;
      for (int l = 0; l < num_labels_; l++) {
        // Current node is (l, i, j).
        edge_score = GetEdgeScore(l, i, j, variable_log_potentials);

        if (j == (i + 1)) {
          chart[l][i][j] = edge_score;
          path[l][i][j] = -1;
        } else {
          // Get the best split and sublabels (k, B, C).
          double best_value;
          int best = -1;
          int best_split;
          int best_left_label;
          int best_right_label;
          // Get the best split.
          for (int k = 0; k < num_labels_; k++) {
            double best_left_value;
            double best_left = -1;
            // Get the best left label.
            for (int b = 0; b < num_labels_; b++) {
              left_val = chart[b][i][k];
              if (best_left < 0 || left_val > best_left_value) {
                best_left_value = left_val;
                best_left = b;
              }
            }
            // Get the best right label.
            double best_right_value;
            double best_right = -1;
            for (int c = 0; c < num_labels_; c++) {
              right_val = chart[c][k][j];
              if (best_right < 0 || right_val > best_right_value) {
                best_right_value = right_val;
                best_right = c;
              }
            }
            // The score of the best expansion.
            double val = edge_score + best_left_val + best_right_val;
            if (best < 0 || val > best_value) {
              best_value = val;
              best = k;
          }
        }
      }
    }
  }


} // namespace AD3

#endif // FACTOR_TREE_H_
