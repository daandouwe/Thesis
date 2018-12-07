# Wrap custom cpp Dynet Expression in python

Say we implemented a custom Dynet Node like the `identity` function in https://github.com/vene/dynet-custom. How then do we wrap this in python, so that we can say
```python
import dynet as dy

>>> dy.identity
<built-in function barfmax>
```
To unravel this, we follow the trail of the python importable function `dynet.sparsemap` and see where this leads us.

This starts in the cython file `_dynet.pyx` where you can see the python sparsemap expression defined as:
```python
cpdef Expression sparsemax(Expression x):
    """Sparsemax

    The sparsemax function (Martins et al. 2016), which is similar to softmax, but induces sparse solutions where most of the vector elements are zero. **Note:** This function is not yet implemented on GPU.

    Args:
        x (dynet.Expression): Input expression

    Returns:
        dynet.Expression: The sparsemax of the scores
    """
    return Expression.from_cexpr(x.cg_version, c_sparsemax(x.c()))
```
The meat of this expression is in the delegation to the function `c_sparsemax(...)`. This is function defined in `dynet/python/_dynet.pxd`. (A pxd file is like a c header file but for cython.)
```python
cdef extern from "dynet/expr.h" namespace "dynet":
    cdef cppclass CExpression "dynet::Expression":
        CExpression()
        CExpression(CComputationGraph *pg, VariableIndex i)
        CComputationGraph *pg
        unsigned i
        CDim dim() except +
        bool is_stale()
        const CTensor& gradient() except +

    cdef enum c_GradientMode "dynet:GradientMode":
        zero_gradient,
        straight_through_gradient
    .
    .
    .

    CExpression c_sparsemax "dynet::sparsemax" (CExpression& x) except + #
```
This looks like magic, until you realise that in the file `dynet/expr.h` (from which de external cdef is made) you can find it defined as
```cpp
Expression sparsemax(const Expression& x);
```
and this is fleshed out in `dynet/expr.cc` where you can read
```cpp
Expression sparsemax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sparsemax>({x.i})); }
```
Again we are delegating all the work to yet some other expression called `Sparsemax` which, upon inspection, is defined in `dynet/nodes.h`. `dynet/nodes.h` simply includes a whole lot of other header files, among which is `dynet/nodes-softmaxes.h`. In `dynet/nodes-softmaxes.h` we finally recognize a the definition of a Dynet Node:
```cpp
struct Sparsemax : public Node {
  explicit Sparsemax(const std::initializer_list<VariableIndex>& a) : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
};
```
This is fleshed out in `dynet/nodes-softmaxes.cc` with all the elements that we recognize from the toy example in `vene/dynet-custom`:
```cpp
// ************* Sparsemax *************

#define MAX_SPARSEMAX_LOSS_ROWS 65536

#ifndef __CUDACC__

string Sparsemax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparsemax(" << arg_names[0] << ")";
  return s.str();
}

Dim Sparsemax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && LooksLikeVector(xs[0]), "Bad input dimensions in Sparsemax: " << xs);
  return xs[0];
}

size_t Sparsemax::aux_storage_size() const {
  return (dim.size() + 1) * sizeof(float);
}

#endif

template<class MyDevice>
void Sparsemax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("Sparsemax forward");
#else
    const unsigned rows = xs[0]->d.rows();
    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    unsigned k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    auto y = mat(fx);
    tvec(fx) = (tvec(*xs[0]) - tau).cwiseMax(0.f);
    int c = 1;
    int *cc = static_cast<int*>(aux_mem);
    for (unsigned i = 0; i < rows; ++i)
      if (y(i,0) > 0.f) cc[c++] = i;
    cc[0] = c - 1;
#endif
  } else {
    DYNET_RUNTIME_ERR("Sparsemax not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void Sparsemax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("Sparsemax backward");
#else
  const int ssize = static_cast<int*>(aux_mem)[0];
  int *support = static_cast<int*>(aux_mem) + 1;
  float dhat = 0;
  auto& d = mat(dEdf);
  for (int i = 0; i < ssize; ++i)
    dhat += d(support[i], 0);
  dhat /= ssize;
  for (int i = 0; i < ssize; ++i)
    (mat(dEdxi))(support[i], 0) += d(support[i], 0) - dhat;
#endif
}
DYNET_NODE_INST_DEV_IMPL(Sparsemax)
```
