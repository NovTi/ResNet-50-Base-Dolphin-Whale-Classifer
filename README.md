## ResNet 50 base Dolphin-Whale Classifer

## Datset
I use the dataset originally from [“Happywhale - Whale and Dolphin Identification”](https://www.kaggle.com/competitions/happy-whale-and-dolphin/overview) Kaggle competition. For a more structured and easy-to-implement data, I found the [resized whale-dolphin images dataset](https://www.kaggle.com/datasets/rdizzl3/jpeg-happywhale-128x128) (contributed by “RDIZZL3”). The contributor resized the images to 128×128.
It contains 3k training images and 1k testing images with two species: bottlenose dolphin and killer whale.


## Further Modification
Want to use the ResNet50 as the backbone of ProtoNet and see its performace.

## Citation
```
@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
```
