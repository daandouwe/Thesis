# python
pip install virtualenv
virtualenv -p python3 ~/stochastic-decoder-env
source ~/stochastic-decoder-env/bin/activate
pip install mxnet-cu"${CUDA}"==1.0.0.post4 sphinx pyyaml typing sphinx tensorboard==1.0.0a6
python setup.py install
sed -i "s@PWD@$PWD@" workflow/sockeye.tconf
deactivate
