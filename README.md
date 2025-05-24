# CNN-BFS
Semantic segmentation-based background-foreground separation on Chicago traffic camera images using a convolutional neural network.

Uses a three-layer fully-connected convolutional neural network in PyTorch. Performance measured using Intersection over Union (IoU) metrics. The algorithm will separate cars from the background landscape in traffic camera images of various resolutions, vehicle densities, and perspectives.

Optimizer: Adam (1e-4 learning rate)
Loss Criterion: BCE With Logits

Performed on Chicago traffic camera images in a dataset organized hierarchically, which is parsed per-scene in the BFSDataset() code.

Access training data here: https://www.zipshare.com/download/e2FyY2hpdmVJZDowMzI2YzdjMy04M2QyLTRhYzAtYTc5Yy01MTdkZTkzZjk4OWN9
