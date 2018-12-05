// Copyright (c) 2018 Daan van Stigt
// All Rights Reserved.

#ifndef FACTOR_TREE
#define FACTOR_TREE

#include "ad3/GenericFactor.h"

namespace AD3 {

// class Arc {
//  public:
//   Arc(int h, int m) : h_(h), m_(m) {}
//   ~Arc() {}
//
//   int head() { return h_; }
//   int modifier() { return m_; }
//
//  private:
//   int h_;
//   int m_;
// };

// A node (l, i, j) is a labeled span
// with label l and span i-j.
class Node{
 public:
  Node(int l, int i, int j) : l_(l), i_(i), j_(j) {}  // I suppose this is assingment of the private variables?
  ~Node() {}
  int label() { return l_; }
  int left() { return i_; }
  int right() { return j_; }

private:
 int l_;
 int i_;
 int j_;
};

// A hyperedge (b:i-k, c:k-j) -> a:i-j connects nodes b and c to a.
// TODO: assert that b.right() == c.left() ?
class Hyperedge {
 public:
  Hyperedge(Node a, Node b, Node c) : a_(a), b_(b), c_(c) {}  // I suppose this is assingment of the private variables?
  ~Hyperedge() {}

  int head() { return a_; }
  int lchild() { return b_; }
  int rchild() { return c_; }

  int head_label() { return a_.label(); }
  int lchild_label() { return b_.label(); }
  int rchild_label() { return c_.label(); }

  int left() { return a_.left(); }
  int right() { return a_.right(); }
  int split() { return b_.right(); }

 private:
  int a_;
  int b_;
  int c_;
};


class FactorTree : public GenericFactor {
 public:
  FactorTree() {}
  virtual ~FactorTree() { ClearActiveSet(); }

  // int RunCLE(const vector<double>& scores,
  //            vector<int> *heads,  // To output predicted heads
  //            double *value);  // To output score
   int RunViterbi(const vector<double>& scores,
                  vector<Node> *node,  // To output predicted nodes. TODO: variable length!
                  double *value);  // To output score

  // // Compute the score of a given assignment.
  // // Note: additional_log_potentials is empty and is ignored.
  // void Maximize(const vector<double> &variable_log_potentials,
  //               const vector<double> &additional_log_potentials,
  //               Configuration &configuration,
  //               double *value) {
  //   vector<int>* heads = static_cast<vector<int>*>(configuration);
  //   RunCLE(variable_log_potentials, heads, value);
  // }

  // Compute the score of a given assignment.
  // Note: additional_log_potentials is empty and is ignored.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    vector<Node>* nodes = static_cast<vector<Node>*>(configuration);
    RunViterbi(variable_log_potentials, nodes, value);
  }

  // // Compute the score of a given assignment.
  // // Note: additional_log_potentials is empty and is ignored.
  // void Evaluate(const vector<double> &variable_log_potentials,
  //               const vector<double> &additional_log_potentials,
  //               const Configuration configuration,
  //               double *value) {
  //   const vector<int> *heads = static_cast<const vector<int>*>(configuration);
  //   // Heads belong to {0,1,2,...}
  //   *value = 0.0;
  //   for (int m = 1; m < heads->size(); ++m) {
  //     int h = (*heads)[m];
  //     int index = index_arcs_[h][m];
  //     *value += variable_log_potentials[index];
  //   }
  // }

  // Compute the score of a given assignment.
  // Note: additional_log_potentials is empty and is ignored.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<Node> *nodes = static_cast<const vector<Node>*>(configuration);
    *value = 0.0;
    for (int n = 1; n < nodes->size(); ++n) {
      Node node = (*nodes)[n];
      int l = node.label();
      int i = node.left();
      int j = node.right();
      int index = index_edges_[l][i][j];
      *value += variable_log_potentials[index];
    }
  }

  // // Given a configuration with a probability (weight),
  // // increment the vectors of variable and additional posteriors.
  // // Note: additional_log_potentials is empty and is ignored.
  // void UpdateMarginalsFromConfiguration(
  //   const Configuration &configuration,
  //   double weight,
  //   vector<double> *variable_posteriors,
  //   vector<double> *additional_posteriors) {
  //   const vector<int> *heads = static_cast<const vector<int>*>(configuration);
  //   for (int m = 1; m < heads->size(); ++m) {
  //     int h = (*heads)[m];
  //     int index = index_arcs_[h][m];
  //     (*variable_posteriors)[index] += weight;
  //   }
  // }
  // Given a configuration with a probability (weight),
  // increment the vectors of variable and additional posteriors.
  // Note: additional_log_potentials is empty and is ignored.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    // const vector<int> *heads = static_cast<const vector<int>*>(configuration);
    const vector<Node> *nodes = static_cast<const vector<Node>*>(configuration);  // What is this??

    for (int n = 1; n < nodes->size(); ++n) {
      Node node = (*nodes)[n];
      int l = node.label();
      int i = node.left();
      int j = node.right();
      int index = index_edges_[l][i][j];
      (*variable_posteriors)[index] += weight;
    }
  }

  // // Count how many common values two configurations have.
  // int CountCommonValues(const Configuration &configuration1,
  //                       const Configuration &configuration2) {
  //   const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
  //   const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
  //   int count = 0;
  //   for (int i = 1; i < heads1->size(); ++i) {
  //     if ((*heads1)[i] == (*heads2)[i]) {
  //       ++count;
  //     }
  //   }
  //   return count;
  // }
  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<Node> *nodes1 = static_cast<const vector<Node>*>(configuration1);
    const vector<Node> *nodes2 = static_cast<const vector<Node>*>(configuration2);
    Node node1, node2;
    bool label, left, right;
    int count = 0;
    for (int i = 1; i < nodes1->size(); ++i) {
      node1 = (*nodes1)[i];
      for (int j = 1; j < nodes2->size(); ++j) {
        node2 = (*nodes2)[j];
        label = (node1.label() == node2.label());
        left = (node1.left() == node2.left());
        right = (node1.right() == node2.right());
        if (label && left && right) {
          ++count;
        }
      }
    }
    return count;
  }

  // Check if two configurations are the same.
  // bool SameConfiguration(
  //   const Configuration &configuration1,
  //   const Configuration &configuration2) {
  //   const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
  //   const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
  //   for (int i = 1; i < heads1->size(); ++i) {
  //     if ((*heads1)[i] != (*heads2)[i]) return false;
  //   }
  //   return true;
  // }
  // Check if two configurations are the same.
  bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<Node> *nodes1 = static_cast<const vector<Node>*>(configuration1);
    const vector<Node> *nodes2 = static_cast<const vector<Node>*>(configuration2);
    if nodes1->size() != nodes2->size() {
      return false;
    }
    for (int i = 1; i < nodes1->size(); ++i) {
      if ((*nodes1)[i] != (*nodes2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  // void DeleteConfiguration(
  //   Configuration configuration) {
  //   vector<int> *heads = static_cast<vector<int>*>(configuration);
  //   delete heads;
  // }
  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<Node> *nodes = static_cast<vector<Node>*>(configuration);
    delete nodes;
  }

  // // Create configuration.
  // Configuration CreateConfiguration() {
  //   vector<int>* heads = new vector<int>(length_);
  //   return static_cast<Configuration>(heads);
  // }
  // Create configuration.
  Configuration CreateConfiguration() {
    // TODO: this is variable-length...
    // A binary tree has n-1 nodes where n is the
    // sentence-length, but our trees are not (perfectly) binary.
    vector<Node>* nodes = new vector<Node>(length_ - 1);
    return static_cast<Configuration>(nodes);
  }

  // public:
  //  void Initialize(int length, const vector<Arc*> &arcs) {
  //    length_ = length;
  //    index_arcs_.assign(length, vector<int>(length, -1));
  //    for (int k = 0; k < arcs.size(); ++k) {
  //      int h = arcs[k]->head();
  //      int m = arcs[k]->modifier();
  //      index_arcs_[h][m] = k;
  //    }
  //  }

 public:
  void Initialize(int length, int num_labels, const vector<Hyperedges*> &edges) {
    length_ = length;
    num_labels_ = num_labels;
    index_edges_.assign(num_labels, length, length);  // TODO: this is incorrect (use resize? see FactorSequence.h)
    for (int k = 0; k < edges.size(); ++k) {
      int l = edges[k]->label();
      int i = edges[k]->left();
      int j = edges[k]->right();
      index_edges_[l][i][j] = k;
    }
  }

 protected:
  // Sentence length.
  int length_;
  // Number of nonterminal labels.
  int num_labels_;
  // At each position, map from hyperedges to a global index which
  // matches the index of additional_log_potentials_.
  vector<vector<vector<int> > > > index_edges_;
};

} // namespace AD3

#endif // FACTOR_TREE
