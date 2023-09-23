# migrate to the new version of the code

## overview

This is the code for the paper "Temporally-Aware Pooling for Relation Extraction with Knowledge Bases" (EMNLP 2018).
This is the project for refactoring and migrating Pytorch code to PytorchLightning.

## why refactoring

- Current code is provided by the owner of the soccernet DVC challenge 2023. It is not easy to use and not easy to debug. It is not easy to reproduce the results. And also pretrained weights from website cannnot be available. Since the fact that both weights and code are no longer operational, I decided to rewrite the code.
- PytorchLightning is a lightweight wrapper for Pytorch. It is easy to use and easy to debug.
- easy to leverage distributed training. PytorchLightning supports distributed training. It is easy to use multiple GPUs to train the model.
- For my training to write more efficiently and more readable code.

## project's goal

- [ ] rewrite the code using PytorchLightning
- [ ] config hydra integration
- [ ] usable pretrained weights
- [ ] runnable training and inference
- [ ] runnable ditributed training
- [ ] increase the readability of the code and my ability to write code

```
lib
├── main.py
├── dataset.py
├── model
│   ├── __init__.py
│   ├── base_model.py
│   ├── captioning.py
│   ├── netvlad.py
│   └── spotting.py
├── trainer.py
├── evaluation.py
├── util.py
└── demo.py
```

## work to do

- [ ] introduce pytest and isort and black, mypy
- [ ] introduce hydra, pytorch-ligihning
- [ ] refactor dataset.py

## やっぱやめる

- train.py に書いている内容が LightNingModule に入って、結構なコード量になる。model と train を分けたいから、ちょっと微妙
- 実装の練習/学習の高速化なら、pyotorch lightning じゃなくてよくね？
