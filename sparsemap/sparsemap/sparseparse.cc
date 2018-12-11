/* SparseMAP in configuration-space with an labeled-span tree factor.
 * Author: Daan van Stigt
 * License: Apache 2.0
 */

#include <cmath>
#include <dynet/nodes-impl-macros.h>
#include <ad3/FactorGraph.h>

#include "sparseparse.h"
#include "FactorTree.h"


using namespace dynet;
using std::string;
using std::vector;
// using sparsemap::FactorTree;

namespace ei = Eigen;

// size_t SparseMST::aux_storage_size() const
// {
//     return (sizeof(int) * ((length_ * max_iter_) + 1) +
//             sizeof(float) * (max_iter_ + 1) * (max_iter_ + 1));
// }
// TODO: is this correct? num_nodes_ instead of length_?
// That is not the size of one configuration...
size_t SparseParse::aux_storage_size() const
{
    return (sizeof(int) * ((num_nodes_ * max_iter_) + 1) +
            sizeof(float) * (max_iter_ + 1) * (max_iter_ + 1));
}

/* very evil pointer arithmetic. look away for 3 rows */
// float* SparseMST::get_inv_A_ptr() const {
//     int* aux = static_cast<int*>(aux_mem);
//     return static_cast<float*>(
//         static_cast<void*>(1 + aux + (length_ * max_iter_)));
// }
float* SparseParse::get_inv_A_ptr() const {
    // A_inv is stored at the end of the auxilliary memory?
    int* aux = static_cast<int*>(aux_mem);
    return static_cast<float*>(
        static_cast<void*>(1 + aux + (num_nodes_ * max_iter_)));
}


int SparseParse::get_n_active() const {
    // Apparently, the first entry of aux_mem contains the size
    // of the memory (number of active configurations in this case).
    int* aux = static_cast<int*>(aux_mem);
    return aux[0];
}


/* TODO: can we return the vector by reference here and preserve it */
// vector<int> SparseMST::get_config(size_t i) const
// {
//     int* active_set_ptr = 1 + static_cast<int*>(aux_mem);
//     return vector<int>(active_set_ptr + length_ * i,
//                        active_set_ptr + length_ * (i + 1));
// }
vector<int> SparseParse::get_config(size_t i) const
{
    // NOTE: aux_mem is a Dynet variable that can be requested always (availlable globally?)
    // NOTE: the first integer holds the size of the memory.
    int* active_set_ptr = 1 + static_cast<int*>(aux_mem);
    return vector<int>(active_set_ptr + num_nodes_ * i,
                       active_set_ptr + num_nodes_ * (i + 1));
}

/**
 * \brief Solves the SparseMAP optimization problem for tree factor
 * \param x ParseNode potentials (= edge potentials!) in topological order
 * \param n_active Location where support size will be stored
 * \param active_set_ptr Location where the active set will be stored
 * \param distribution_ptr Location where the posterior weights will be stored
 * \param inverse_A_ptr Location where (MtM)^-1 will be stored
 */

void SparseParse::sparsemap_decode(const Tensor* x,
                                   int* n_active,
                                   int* active_set_ptr,
                                   float* distribution_ptr,
                                   float* inverse_A_ptr) const
{
    auto xvec = vec(*x);

    // run ad3
    AD3::FactorGraph factor_graph;
    // factor_graph.SetVerbosity(3);
    vector<AD3::BinaryVariable*> vars;
    // The nodes that will be used in Initialize
    vector<AD3::ParseNode*> nodes;

    // Instantiate variable for each arc.
    // for (int m = 1; m < length_; ++m)
    //     for (int h = 0; h < length_; ++h)
    //         if (h != m)
    //             vars.push_back(factor_graph.CreateBinaryVariable());

    // Instantiate a variable for each node in topological order.
    // Given that the factor for an edge is only dependent on the head
    // node of the edge we require only one log-potential for each node.
    // See FactorTree::RunViterbi for a detailed illustration.
    for (int s = 1; s < length_ + 1; ++s) {
      for (int i = 0; i < length_ + 1 - s; i++) {
        int j = i + s;  // span of length s from i to j
        for (int l = 0; l < num_labels_; l++) {
          AD3::ParseNode *node = new AD3::ParseNode(l, i, j);
          nodes.push_back(node);
          vars.push_back(factor_graph.CreateBinaryVariable());
          // TODO: figure out, do we need more binary variables?? E.g. one for each edge?
        }
      }
    }

    std::cout << "sparseparse.cc >> Size of nodes: " << nodes.size() << std::endl;

    // num_nodes_ = nodes.size()

    // input potentials
    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());  // WHY xvec.data() + xvec.size() ???

    std::cout << "sparseparse.cc >> Size of input potentials: " << unaries_in.size() << std::endl;
    std::cout << "sparseparse.cc >> Data of input potentials: ";
    for (auto const& c : unaries_in)
      std::cout << c << ' ';
    std::cout << std::endl;

    // output variables (additionals will be unused)
    vector<double> unaries_post;
    vector<double> additionals;
    // FactorTree tree_factor;
    AD3::FactorTree tree_factor;

    factor_graph.DeclareFactor(&tree_factor, vars, false);
    tree_factor.SetQPMaxIter(max_iter_);
    tree_factor.SetClearCache(false);
    tree_factor.Initialize(length_, num_labels_, nodes);
    tree_factor.SolveQP(unaries_in, additionals, &unaries_post, &additionals);

    auto active_set = tree_factor.GetQPActiveSet();
    *n_active = active_set.size();

    // write distribution as floats
    auto distribution = tree_factor.GetQPDistribution();
    assert(active_set.size() <= distribution.size());
    std::copy(distribution.begin(),
              distribution.begin() + *n_active,
              distribution_ptr);

    // std::cout << "Distribution ";
    // for(auto &&v : distribution) std::cout << v << " ";
    // std::cout << std::endl;

    // TODO: also fix this because some kind of conversion of the
    // configuration is taking place here.
    // NOTE: is this used anywhere???
    // int* active_set_moving_ptr = active_set_ptr;
    // for (auto&& cfg_ptr : active_set)  // active set is of type vector<Configuration>
    // {
    //     auto cfg = static_cast<vector<int>*>(cfg_ptr);  // convert the cfg_ptr to a pointer to heads
    //     std::copy(cfg->begin(), cfg->end(), active_set_moving_ptr);
    //     active_set_moving_ptr += cfg->size();
    // }

    auto inverse_A = tree_factor.GetQPInvA();
    std::copy(inverse_A.begin(), inverse_A.end(), inverse_A_ptr);
}


/**
 * \brief Backward pass restricted to the active set:
 *
 * \param dEdf_bar Gradient dE wrt layer output
 * \param dEdx Gradient wrt potentials x; incremented in place
 *
 * Computes dE/db_bar as intermediate quantity
 * where:
 *
 * f(x) is the sparse distribution over all possible trees
 * f_bar(x) is f restricted to its support
 * b_bar(x) is the vector of total scores for the configurations in the support
 *
 */
void SparseParse::backward_restricted(const eivector& dEdf_bar, Tensor& dEdx)
const
{

    int* aux = static_cast<int*>(aux_mem);
    int* active_set_ptr = 1 + aux;
    float* inv_A_ptr = get_inv_A_ptr();

    int n_active = *aux;

    ei::Map<ei::Matrix<float, ei::Dynamic, ei::Dynamic, ei::RowMajor> >
        e_inva(inv_A_ptr, 1 + n_active, 1 + n_active);

    /* A^-1    = [ k   b^T]
     *           [ b    S ]
     *
     * (MtM)^-1 = S - (1/k) outer(b, b)
     */

    float k = e_inva(0, 0);
    auto b = e_inva.row(0).tail(n_active);
    auto S = e_inva.bottomRightCorner(n_active, n_active);
    S.noalias() -= (1 / k) * (b.transpose() * b);
    auto first_term = dEdf_bar * S;
    auto second_term = (first_term.sum() * S.rowwise().sum()) / S.sum();
    auto dEdb_bar = first_term - second_term.transpose();

    // TODO!
    // map gradients back to the unaries that contribute to them
    // (sparse multiplication by M)
    for (int i = 0; i < n_active ; ++i)
    {
        // A configuration is a list of heads. In our case,
        // a configuration is variable length...
        for (int mod = 1; mod < length_; ++mod)
        {
            size_t mod_address = (i * length_) + mod;
            int head = active_set_ptr[mod_address];
            mat(dEdx)(head, mod - 1) += dEdb_bar(i);
        }
    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% //

string SparseParse::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "sparse_parse(" << arg_names[0] << ")";
    return s.str();
}


Dim SparseParse::dim_forward(const vector<Dim> &xs) const
{
    DYNET_ARG_CHECK(xs.size() == 1, "SparseParse takes a single input");
    // DYNET_ARG_CHECK(xs[0] == num_nodes_, "input has wrong first dim");
    unsigned int d = max_iter_;  // TODO: figure out why this here.
    return Dim({d});
}


template<class MyDevice>
void SparseParse::forward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor*>& xs,
                                   Tensor& fx) const
{
    const Tensor* x = xs[0];

    auto out = vec(fx);
    out.setZero();

    int* aux = static_cast<int*>(aux_mem);
    int* active_set_ptr = aux + 1;
    float* inv_A_ptr = get_inv_A_ptr();

    vector<float> distribution(max_iter_);
    sparsemap_decode(x, aux, active_set_ptr, out.data(), inv_A_ptr);
}

template <class MyDevice>
void SparseParse::backward_dev_impl(const MyDevice& dev,
                                    const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const
{
    int n_active = *(static_cast<int*>(aux_mem));
    ei::Map<const eivector> dEdf_bar(mat(dEdf).data(), n_active);
    backward_restricted(dEdf_bar, dEdxi);
}

DYNET_NODE_INST_DEV_IMPL(SparseParse)

Expression sparse_parse(const Expression& x, size_t length, size_t num_labels, size_t max_iter) {
    return Expression(x.pg,
                      x.pg->add_function<SparseParse>({x.i}, length, num_labels, max_iter));
}
