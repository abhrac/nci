# Learning Conditional Invariances through Non-Commutativity
Official implementation of "[Learning Conditional Invariances through Non-Commutativity](https://openreview.net/forum?id=tUVG9nGzgE&noteId=2XWklnrtBv)", ICLR 2024.

The current version provides a non-commutative version of Domain Adversarial Neural Networks. In theory, any invariance learning algorithm that has an associated commutative operator (Definition 2 in the main paper), can be adapted to have an NCI variant.

## Environment Setup

1. Clone the project repository:
```shell
git clone https://github.com/abhrac/nci.git
``` 
2. Install dependences:
```shell
pip install -r requirements.txt
```
3. Run
```shell
python main.py --algorithm=NCI --data_dir=path/to/dataset/root --dataset=PACS --uda_holdout_fraction=0.2 --task=domain_adaptation --batch_size=64
```

## Citation
```
@inproceedings{
  chaudhuri2024nci,
  title={Learning Conditional Invariances through Non-Commutativity},
  author={Abhra Chaudhuri, Serban Georgescu, Anjan Dutta},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=tUVG9nGzgE}
}
```

## Acknowledgements
Experimentation framework adapted from [DomainBed](https://github.com/facebookresearch/DomainBed).
