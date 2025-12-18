# Domain-Adaptation
1. DA: Feature-level Adversarial Alignment Framework

<img width="316" height="285" alt="image" src="https://github.com/user-attachments/assets/d1a97563-c62e-4416-b8af-0bdfffded45a" />

The feature-level advertising alignment is a competitive operation between the domain discriminator and the backbone, resulting in adversarial learning. During the learning process, the domain discriminator separates the source domain and the target domain of the input image so that the model is learned, while the backbone repeats the process of deceiving the domain discriminator from distinguishing the source domain and the target domain during error backpropagation. Eventually, the balance between the domain discriminator and the backbone is balanced to ensure that the backbone features are aligned in a state that is unchanged from the domain.

<img width="456" height="444" alt="image" src="https://github.com/user-attachments/assets/beb5b402-3390-4561-8e44-073db017d34e" />

Through the development of a DANN (Domain Adversarial Neural Network)-based Feature-level Adversarial Alignment pipeline, the feature distribution between the source and target domains cannot be distinguished so that the backbone learns domain-invariant features to enhance the generalization performance of the target domain. Model learning uses the YOLOv8 backbone as a feature extractor and aligns the feature distribution of the source and target domains through a domain discriminator.
In this study, after analyzing the complex neural network structure of YOLOv8, the impact of learning was limited by explicitly separating the backbone-neck path to prevent adversarial updates from being transmitted to the detection head. Multiple intermediate feature maps generated through the backbone path are collected by hook or backbone-only forward operations, and used as input to the domain discriminator so that domain discrimination is based solely on the backbone representation.

In the learning stage, the Source Domain image is used to minimize the task loss of the object detection model and the domain discriminator is learned by combining the Source and Target Domain features. In other words, in forward propagation, the Gradient Reversal Layer operates as an identity function to deliver the backbone feature to the domain discriminator as it is, and while the domain discriminator learns to distinguish between the Source Domain and the Target Domain, the backbone receives gradients in the opposite direction through backpropagation and updates the model and the domain discriminator alternately by learning the domain invariant representation so that the features of the two domains cannot be distinguished.

This structure in which hostile losses are directly reflected in learning backbone feature representations serves as the basis for more stable and meaningful feature-level alignment while fundamentally preventing distortion of detection heads. In addition, since the backbone performs alignment around the layer where high-level features are formed, a robust representation is built against visual changes in images of bad weather. As a result, the model learns based on the Source Domain labels while reducing the distribution difference between domains to improve the Target Domain generalization performance.

<img width="1065" height="669" alt="image" src="https://github.com/user-attachments/assets/5a2d448b-020b-49f3-af6e-732e7ce7991a" />

