# BPR: Bayesian Personalized Ranking from Implicit Feedback

This repository contains a PyTorch implementation of the _BPR: Bayesian Personalized Ranking from Implicit Feedback_ loss, as proposed in the paper:

> Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme. "BPR: Bayesian Personalized Ranking from Implicit Feedback" https://arxiv.org/abs/1205.2618

The implementation is owned by Jeong-Junhwan and includes a simple training process with bpr loss using Matrix Factorization model.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Requirements

To install the required dependencies, run:

```bash
pip3 install -r requirements.txt
```

## Usage

To run the training process, simply execute the following command:

```bash
python3 train.py
```

Please note that this implementation currently only support MF model. You are welcome to contribute by adding these features.

## Dataset

This implementation has been tested using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which is a popular benchmark dataset for collaborative filtering models. It consists of 100,000 ratings from 1 to 5, given by 943 users on 1,682 movies.

To use the ML-100K dataset, please download it from the link above and place the unzipped files in a ml-100k/ directory within the project root.

## Contributing

Contributions are welcome! If you would like to improve this implementation, add new features or fix bugs, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
