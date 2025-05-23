# CG-ODE

CG-ODE is an overall framework for learning the co-evolutioin of nodes and edges in multi-agent dynamical systems.

You can see our KDD 2021 paper [“**Coupled Graph ODE for Learning Interacting System Dynamics**”](https://dl.acm.org/doi/10.1145/3447548.3467385?sid=SCITRUS) for more details.

This implementation of CG-ODE is based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) API.
## Data Preparation

#### Covid-19 Dataset

The daily trendy data is obtained from [JHU CSSE](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us).  The mobility data is obtained from [SafeGraph](https://docs.safegraph.com/docs/weekly-patterns) where you need to register first and request the data. The script for generating the mobility data matrix from raw data is available upon request.

#### Social Network Dataset

Generate the simulated social opinion dynamic dataset by running:

```bash
cd data/social
python generate_socialNetwork.py 
```

The original generation process can be found at  [Co-Evolve KDD17](http://web.cs.ucla.edu/~yzsun/papers/2017_kdd_coevolution.pdf) 



## Setup

This implementation is based on pytorch_geometric. To run the code, you need the following dependencies:

* [Python 3.6.10](https://www.python.org/)

- [Pytorch 1.4.0](https://pytorch.org/)

- [pytorch_geometric 1.4.3](https://pytorch-geometric.readthedocs.io/)

  - torch-cluster==1.5.3
  - torch-scatter==2.0.4
  - torch-sparse==0.6.1

- [torchdiffeq==0.0.1](https://github.com/rtqichen/torchdiffeq)

- [numpy 1.16.1](https://numpy.org/)


