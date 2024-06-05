# nn-playingcards

classify images using [pytorch on apple silicon hardware](https://developer.apple.com/metal/pytorch).

required: [pytorch](https://pytorch.org/) incl. [torchvision](https://pytorch.org/vision), [timm](https://github.com/huggingface/pytorch-image-models) and [mpl](https://matplotlib.org/).

```
pip3 install torch torchvision timm matplotlib
```

download the playingcards [here](https://github.com/xeaydin/Card-Image-Classification/tree/master/Dataset) and adjust `cards_folder` accordingly

## training

run `playingcards_training.py`

training state will be saved to `playingcards_trainstate.pt`

## test image

run `playingcards_classify.py <IMAGEFILE>`
