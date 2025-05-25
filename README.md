# CNN-BFS
Uses a three-layer fully-connected convolutional neural network in PyTorch. Performance measured using Intersection over Union (IoU) metrics. The algorithm will separate cars from the background landscape in traffic camera images of various resolutions, vehicle densities, and perspectives.

Optimizer: Adam (1e-4 learning rate)<br />
Loss Criterion: BCE With Logits

Performed on Chicago traffic camera images in a dataset organized hierarchically, which is parsed per-scene in the BFSDataset() code.

Results:
![Result of image segmentation](https://github.com/user-attachments/assets/304f6699-f504-4418-9a81-b7c9e5c2136b)
**Note**: IoU is performing poorly at the moment and prediction image is noisy; will be fixed in a future commit.

Access training data here: <a href="https://github.com/siddhshah/CNN-BFS/releases/tag/v1.0" download>Download ZIP</a>
- Please edit image and annotation paths in the code accordingly to match the location of the dataset in your filesystem.
