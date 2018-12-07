/* SparseMAP in configuration-space with an MST dependency tree factor.
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#pragma once

#include <Eigen/Eigen>
#include <dynet/dynet.h>
#include <dynet/nodes-def-macros.h>
#include <dynet/expr.h>
#include <dynet/tensor-eigen.h>

#include <map>

using namespace dynet;
using std::string;
using std::vector;

typedef Eigen::RowVectorXf eivector;


struct SparseParse : public Node {

    size_t length_;
    size_t max_iter_;

    explicit SparseParse(const std::initializer_list<VariableIndex>& a,
                       size_t length, size_t num_labels, size_t max_iter)
        : Node(a), length_(length), num_labels_(num_labels), max_iter_(max_iter)
    {
        this->has_cuda_implemented = false;
    }
    DYNET_NODE_DEFINE_DEV_IMPL()
    size_t aux_storage_size() const override;

    int get_n_active() const;
    vector<int> get_config(size_t i) const;

    protected:
    void sparsemap_decode(const Tensor*, int*, int*, float*, float*) const;
    void backward_restricted(const eivector& dEdfbar, Tensor& dEdx) const;

    float* get_inv_A_ptr() const;
};


/**
 * \brief Compute SparseMAP sparse posterior over trees.
 * \param x Input edge scores (variable_log_potentials), but note: one factor per node (l,i,j) in topological order.
 * \param length Number of words in sentence
 * \param num_labels Number of labels in grammar
 * \param max_iter Number of iterations of Active Set to perform.
 * \return sparse vector of posteriors (fixed size = max_iter)
 *
 * To correctly identify which tree each output index corresponds to,
 * use SparseMST::get_config.
 */
Expression sparse_parse(const Expression& x, size_t length, size_t num_labels, size_t max_iter=10);
