# Implementation of AI algorithms

Example of testing command:
`python3 ./classification/svm_improved.py --file_path ./heart.csv --label_name sex --eval_mode`

- Supervised
    - classifications
        - Statistical learning models
            - SVM
            - LR
            - Boosting
            - GBDT
            - kNN
            - NB
        - Deep learning models (implemented using PyTorch)
            - Computer Vision
                - Cls
                    - FFNN
                    - CNN
                    - GoogLeNet (Inception)
                    - ResNet
                    - RNN (for image classifications)
                - Det
                    - YOLO (v3)
                - Generative Models
                    - PixelCNN
                - Others
                    - Neural Style Transfer
            - NLP
                - RNN
                - BERT (TODO)
    - Regressions
        - (TODO)
    - Reinforcement Learning
        - Q-learning
        - Double Q-learning
        - Sarsa (TODO)
        - DQN
        - A2C (TODO)
        - DDPG (TODO)
        - PPO (TODO)
- Unsupervised
    - Classic
        - kMeans
        - Gaussian Process Regressor
        - Bayesian Optimization
        - Apriori (TODO)
    - Deep Learning
        - Adverserial (TODO)
        - Distillation (TODO)