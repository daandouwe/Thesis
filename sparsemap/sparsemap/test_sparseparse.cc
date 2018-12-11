#include <dynet/dynet.h>
#include <dynet/expr.h>

#include <iostream>

#include "sparseparse.h"

namespace dy = dynet;


int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    const unsigned LENGTH = 5;
    const unsigned NUM_LABELS = 8;

    int num_nodes = 0;
    for (int s = 1; s < LENGTH + 1; ++s) {
      for (int i = 0; i < LENGTH + 1 - s; i++) {
        // int j = i + s;  // span of length s from i to j
        for (int l = 0; l < NUM_LABELS; l++) {
          num_nodes += 1;
        }
      }
    }

    const unsigned NUM_NODES = static_cast<unsigned int>(num_nodes);

    std::cout << num_nodes << std::endl;

    dy::ParameterCollection m;
    Parameter x_par = m.add_parameters({NUM_NODES}, 0, "x");
    ComputationGraph cg;
    Expression x = dy::parameter(cg, x_par);
    Expression y = sparse_parse(x, LENGTH, NUM_LABELS);
    std::cout << y.value() << std::endl;

    auto sparse_parse_node = static_cast<SparseParse*>(cg.nodes[y.i]);
    int n_active = sparse_parse_node->get_n_active();
    std::cout << n_active << std::endl;
    for (int k = 0; k < n_active; ++k) {
        for(auto&& hd : sparse_parse_node->get_config(k))
            std::cout << hd << " ";
        std::cout << std::endl;
    }
}
