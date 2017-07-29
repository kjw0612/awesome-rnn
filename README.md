# Awesome Recurrent Neural Networks

A curated list of resources dedicated to recurrent neural networks (closely related to *deep learning*).

Maintainers - [Myungsub Choi](https://github.com/myungsub), [Taeksoo Kim](https://github.com/jazzsaxmafia), [Jiwon Kim](https://github.com/kjw0612)

We have pages for other topics: [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-random-forest](https://github.com/kjw0612/awesome-random-forest)


## Contributing
Please feel free to [pull requests](https://github.com/kjw0612/awesome-rnn/pulls), email Myungsub Choi (cms6539@gmail.com) or join our chats to add links.

[![Join the chat at https://gitter.im/kjw0612/awesome-rnn](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/kjw0612/awesome-rnn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Sharing
+ [Share on Twitter](http://twitter.com/home?status=http://jiwonkim.org/awesome-rnn%0AResources%20for%20Recurrent%20Neural%20Networks)
+ [Share on Facebook](http://www.facebook.com/sharer/sharer.php?u=https://jiwonkim.org/awesome-rnn)
+ [Share on Google Plus](http://plus.google.com/share?url=https://jiwonkim.org/awesome-rnn)
+ [Share on LinkedIn](http://www.linkedin.com/shareArticle?mini=true&url=https://jiwonkim.org/awesome-rnn&title=Awesome%20Recurrent%20Neural&Networks&summary=&source=)

## Table of Contents

- [Codes](#codes)
- [Theory](#theory)
  - [Lectures](#lectures)
  - [Books / Thesis](#books--thesis)
  - [Architecture Variants](#architecture-variants)
    - [Structure](#structure)
    - [Memory](#memory)
  - [Surveys](#surveys)
- [Applications](#applications)
  - [Natural Language Processing](#natural-language-processing)
    - [Language Modeling](#language-modeling)
    - [Speech Recognition](#speech-recognition)
    - [Machine Translation](#machine-translation)
    - [Conversation Modeling](#conversation-modeling)
    - [Question Answering](#question-answering)
  - [Computer Vision](#computer-vision)
    - [Object Recognition](#object-recognition)
    - [Image Generation](#image-generation)
    - [Video Analysis](#video-analysis)
  - [Multimodal (CV+NLP)](#multimodal-cv--nlp)
    - [Image Captioning](#image-captioning)
    - [Video Captioning](#video-captioning)
    - [Visual Question Answering](#visual-question-answering)
  - [Turing Machines](#turing-machines)
  - [Robotics](#robotics)
  - [Other](#other)
- [Datasets](#datasets)
- [Blogs](#blogs)
- [Online Demos](#online-demos)

## Codes
* [Tensorflow](https://www.tensorflow.org/) - Python, C++
  * [Get started](https://www.tensorflow.org/versions/master/get_started/index.html), [Tutorials](https://www.tensorflow.org/versions/master/tutorials/index.html)
    * [Recurrent Neural Network Tutorial](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html)
    * [Sequence-to-Sequence Model Tutorial](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html)
  * [Tutorials](https://github.com/nlintz/TensorFlow-Tutorials) by nlintz
  * [Notebook examples](https://github.com/aymericdamien/TensorFlow-Examples) by aymericdamien
  * [Scikit Flow (skflow)](https://github.com/tensorflow/skflow) - Simplified Scikit-learn like Interface for TensorFlow
  * [Keras](http://keras.io/) : (Tensorflow / Theano)-based modular deep learning library similar to Torch
  * [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) by sherjilozair: char-rnn in tensorflow
  * [tensorflow-char-rnn](https://github.com/crazydonkey200/tensorflow-char-rnn) by crazydonkey200: char-rnn in tensorflow with customizable parameters
* [Theano](http://deeplearning.net/software/theano/) - Python
  * Simple IPython [tutorial on Theano](http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb)
  * [Deep Learning Tutorials](http://www.deeplearning.net/tutorial/)
    * [RNN for semantic parsing of speech](http://www.deeplearning.net/tutorial/rnnslu.html#rnnslu)
    * [LSTM network for sentiment analysis](http://www.deeplearning.net/tutorial/lstm.html#lstm)
  * [Pylearn2](http://deeplearning.net/software/pylearn2/) : Library that wraps a lot of models and training algorithms in deep learning
  * [Blocks](https://github.com/mila-udem/blocks) : modular framework that enables building neural network models
  * [Keras](http://keras.io/) : (Tensorflow / Theano)-based modular deep learning library similar to Torch
  * [Lasagne](https://github.com/Lasagne/Lasagne) : Lightweight library to build and train neural networks in Theano
  * [theano-rnn](https://github.com/gwtaylor/theano-rnn) by Graham Taylor
  * [Passage](https://github.com/IndicoDataSolutions/Passage) : Library for text analysis with RNNs
  * [Theano-Lights](https://github.com/Ivaylo-Popov/Theano-Lights) : Contains many generative models
* [Caffe](https://github.com/BVLC/caffe) - C++ with MATLAB/Python wrappers
  * [LRCN](http://jeffdonahue.com/lrcn/) by Jeff Donahue
* [Torch](http://torch.ch/) - Lua
  * [torchnet](https://github.com/torchnet/torchnet) : modular framework that enables building neural network models
  * [char-rnn](https://github.com/karpathy/char-rnn) by Andrej Karpathy : multi-layer RNN/LSTM/GRU for training/sampling from character-level language models
  * [torch-rnn](https://github.com/jcjohnson/torch-rnn) by Justin Johnson : reusable RNN/LSTM modules for torch7 - much faster and memory efficient reimplementation of char-rnn
  * [neuraltalk2](https://github.com/karpathy/neuraltalk2) by Andrej Karpathy : Recurrent Neural Network captions image, much faster and better version of the original [neuraltalk](https://github.com/karpathy/neuraltalk)
  * [LSTM](https://github.com/wojzaremba/lstm) by Wojciech Zaremba : Long Short Term Memory Units to train a language model on word level Penn Tree Bank dataset
  * [Oxford](https://github.com/oxford-cs-ml-2015) by Nando de Freitas : Oxford Computer Science - Machine Learning 2015 Practicals
  * [rnn](https://github.com/Element-Research/rnn) by Nicholas Leonard : general library for implementing RNN, LSTM, BRNN and BLSTM (highly unit tested).
* [PyTorch](http://pytorch.org/) - Python
  * [Word-level RNN example](https://github.com/pytorch/examples/tree/master/word_language_model) : demonstrates PyTorch's built in RNN modules for language modeling
  * [Practical PyTorch tutorials](https://github.com/spro/practical-pytorch) by Sean Robertson : focuses on using RNNs for Natural Language Processing
  * [Deep Learning For NLP In PyTorch](https://github.com/rguthrie3/DeepLearningForNLPInPytorch) by Robert Guthrie : written for a Natural Language Processing class at Georgia Tech
* [DL4J](http://deeplearning4j.org/) by [Skymind](http://www.skymind.io/) : Deep Learning library for Java, Scala & Clojure on Hadoop, Spark & GPUs
  * [Documentation](http://deeplearning4j.org/) (Also in [Chinese](http://deeplearning4j.org/zh-index.html), [Japanese](http://deeplearning4j.org/ja-index.html), [Korean](http://deeplearning4j.org/kr-index.html)) : [RNN](http://deeplearning4j.org/usingrnns.html), [LSTM](http://deeplearning4j.org/lstm.html)
  * [rnn examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent)
* Etc.
  * [Neon](http://neon.nervanasys.com/docs/latest/index.html): new deep learning library in Python, with support for RNN/LSTM, and a fast image captioning model
  * [Brainstorm](https://github.com/IDSIA/brainstorm): deep learning library in Python, developed by IDSIA, thereby including various recurrent structures
  * [Chainer](http://chainer.org/) : new, flexible deep learning library in Python
  * [CGT](http://joschu.github.io/)(Computational Graph Toolkit) : replicates Theano's API, but with very short compilation time and multithreading
  * [RNNLIB](https://sourceforge.net/p/rnnl/wiki/Home/) by Alex Graves : C++ based LSTM library
  * [RNNLM](http://rnnlm.org/) by Tomas Mikolov : C++ based simple code
  * [faster-RNNLM](https://github.com/yandex/faster-rnnlm) of Yandex : C++ based rnnlm implementation aimed to handle huge datasets
  * [neuraltalk](https://github.com/karpathy/neuraltalk) by Andrej Karpathy : numpy-based RNN/LSTM implementation
  * [gist](https://gist.github.com/karpathy/587454dc0146a6ae21fc) by Andrej Karpathy : raw numpy code that implements an efficient batched LSTM
  * [Recurrentjs](https://github.com/karpathy/recurrentjs) by Andrej Karpathy : a beta javascript library for RNN
  * [DARQN](https://github.com/5vision/DARQN) by 5vision : Deep Attention Recurrent Q-Network

## Theory
### Lectures
* Stanford NLP ([CS224d](http://cs224d.stanford.edu/index.html)) by Richard Socher
  * [Lecture Note 3](http://cs224d.stanford.edu/lecture_notes/LectureNotes3.pdf) : neural network basics
  * [Lecture Note 4](http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf) : RNN language models, bi-directional RNN, GRU, LSTM
* Stanford vision ([CS231n](http://cs231n.github.io/)) by Andrej Karpathy
  * About NN basic, and CNN
* Oxford [Machine Learning](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) by Nando de Freitas
  * [Lecture 12](https://www.youtube.com/watch?v=56TYLaQN4N8) : Recurrent neural networks and LSTMs
  * [Lecture 13](https://www.youtube.com/watch?v=-yX1SYeDHbg) : (guest lecture) Alex Graves on Hallucination with RNNs

### Books / Thesis
* Alex Graves (2008)
  * [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
* Tomas Mikolov (2012)
  * [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
* Ilya Sutskever (2013)
  * [Training Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
* Richard Socher (2014)
  * [Recursive Deep Learning for Natural Language Processing and Computer Vision](http://nlp.stanford.edu/~socherr/thesis.pdf)
* Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  * [The Deep Learning Book chapter 10](http://www.deeplearningbook.org/contents/rnn.html)


### Architecture Variants

#### Structure

* Bi-directional RNN [[Paper](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]
  * Mike Schuster and Kuldip K. Paliwal, *Bidirectional Recurrent Neural Networks*, Trans. on Signal Processing 1997
* Multi-dimensional RNN [[Paper](http://arxiv.org/pdf/0705.2011.pdf)]
  * Alex Graves, Santiago Fernandez, and Jurgen Schmidhuber, *Multi-Dimensional Recurrent Neural Networks*, ICANN 2007
* GFRNN [[Paper-arXiv](http://arxiv.org/pdf/1502.02367)] [[Paper-ICML](http://jmlr.org/proceedings/papers/v37/chung15.pdf)] [[Supplementary](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)]
  * Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio, *Gated Feedback Recurrent Neural Networks*, arXiv:1502.02367 / ICML 2015
* Tree-Structured RNNs
  * Kai Sheng Tai, Richard Socher, and Christopher D. Manning, *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks*, arXiv:1503.00075 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.00075)]
  * Samuel R. Bowman, Christopher D. Manning, and Christopher Potts, *Tree-structured composition in neural networks without tree-structured architectures*, arXiv:1506.04834 [[Paper](http://arxiv.org/pdf/1506.04834)]
* Grid LSTM [[Paper](http://arxiv.org/pdf/1507.01526)] [[Code](https://github.com/coreylynch/grid-lstm)]
  * Nal Kalchbrenner, Ivo Danihelka, and Alex Graves, *Grid Long Short-Term Memory*, arXiv:1507.01526
* Segmental RNN [[Paper](http://arxiv.org/pdf/1511.06018v2.pdf)]
  * Lingpeng Kong, Chris Dyer, Noah Smith, "Segmental Recurrent Neural Networks", ICLR 2016.
* Seq2seq for Sets [[Paper](http://arxiv.org/pdf/1511.06391v4.pdf)]
  * Oriol Vinyals, Samy Bengio, Manjunath Kudlur, "Order Matters: Sequence to sequence for sets", ICLR 2016.
* Hierarchical Recurrent Neural Networks [[Paper](http://arxiv.org/abs/1609.01704)]
  * Junyoung Chung, Sungjin Ahn, Yoshua Bengio, "Hierarchical Multiscale Recurrent Neural Networks", arXiv:1609.01704

#### Memory

* LSTM [[Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)]
  * Sepp Hochreiter and Jurgen Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997
* GRU (Gated Recurrent Unit) [[Paper](http://arxiv.org/pdf/1406.1078.pdf)]
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014
* NTM [[Paper](http://arxiv.org/pdf/1410.5401)]
  * A.Graves, G. Wayne, and I. Danihelka., *Neural Turing Machines,* arXiv preprint arXiv:1410.5401
* Neural GPU [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]
  * Łukasz Kaiser, Ilya Sutskever, arXiv:1511.08228 / ICML 2016 (under review)
* Memory Network [[Paper](http://arxiv.org/pdf/1410.3916)]
  * Jason Weston, Sumit Chopra, Antoine Bordes, *Memory Networks,* arXiv:1410.3916
* Pointer Network [[Paper](http://arxiv.org/pdf/1506.03134)]
  * Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly, *Pointer Networks*, arXiv:1506.03134 / NIPS 2015
* Deep Attention Recurrent Q-Network [[Paper](http://arxiv.org/abs/1512.01693)]
  * Ivan Sorokin, Alexey Seleznev, Mikhail Pavlov, Aleksandr Fedorov, Anastasiia Ignateva, *Deep Attention Recurrent Q-Network* , arXiv:1512.01693
* Dynamic Memory Networks [[Paper](http://arxiv.org/abs/1506.07285)]
  * Ankit Kumar, Ozan Irsoy, Peter Ondruska, Mohit Iyyer, James Bradbury, Ishaan Gulrajani, Victor Zhong, Romain Paulus, Richard Socher, "Ask Me Anything: Dynamic Memory Networks for Natural Language Processing", arXiv:1506.07285

### Surveys
* Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, [Deep Learning](http://www.nature.com/nature/journal/v521/n7553/pdf/nature14539.pdf), Nature 2015
* Klaus Greff, Rupesh Kumar Srivastava, Jan Koutnik, Bas R. Steunebrink, Jurgen Schmidhuber, [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069), arXiv:1503.04069
* Zachary C. Lipton, [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/pdf/1506.00019), arXiv:1506.00019
* Andrej Karpathy, Justin Johnson, Li Fei-Fei, [Visualizing and Understanding Recurrent Networks](http://arxiv.org/pdf/1506.02078), arXiv:1506.02078
* Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever, [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf), ICML, 2015.

## Applications

### Natural Language Processing

#### Language Modeling
* Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Recurrent Neural Network based Language Model*, Interspeech 2010 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)]
* Tomas Mikolov, Stefan Kombrink, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Extensions of Recurrent Neural Network Language Model*, ICASSP 2011 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)]
* Stefan Kombrink, Tomas Mikolov, Martin Karafiat, Lukas Burget, *Recurrent Neural Network based Language Modeling in Meeting Recognition*, Interspeech 2011 [[Paper](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)]
* Jiwei Li, Minh-Thang Luong, and Dan Jurafsky, *A Hierarchical Neural Autoencoder for Paragraphs and Documents*, ACL 2015 [[Paper](http://arxiv.org/pdf/1506.01057)], [[Code](https://github.com/jiweil/Hierarchical-Neural-Autoencoder)]
* Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, and Richard S. Zemel, *Skip-Thought Vectors*, arXiv:1506.06726 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.06726.pdf)]
* Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush, *Character-Aware Neural Language Models*, arXiv:1508.06615 [[Paper](http://arxiv.org/pdf/1508.06615)]
* Xingxing Zhang, Liang Lu, and Mirella Lapata, *Tree Recurrent Neural Networks with Application to Language Modeling*, arXiv:1511.00060 [[Paper](http://arxiv.org/pdf/1511.00060.pdf)]
* Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston, *The Goldilocks Principle: Reading children's books with explicit memory representations*, arXiv:1511.0230 [[Paper](http://arxiv.org/pdf/1511.02301.pdf)]


#### Speech Recognition
* Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, and Brian Kingsbury, *Deep Neural Networks for Acoustic Modeling in Speech Recognition*, IEEE Signam Processing Magazine 2012 [[Paper](http://cs224d.stanford.edu/papers/maas_paper.pdf)]
* Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton, *Speech Recognition with Deep Recurrent Neural Networks*, arXiv:1303.5778 / ICASSP 2013 [[Paper](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)]
* Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, *Attention-Based Models for Speech Recognition*, arXiv:1506.07503 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.07503)]
* Haşim Sak, Andrew Senior, Kanishka Rao, and Françoise Beaufays. *Fast and Accurate Recurrent Neural Network Acoustic Models for Speech Recognition*, arXiv:1507.06947 2015 [[Paper](http://arxiv.org/pdf/1507.06947v1.pdf)].

#### Machine Translation
* Oxford [[Paper](http://www.nal.ai/papers/kalchbrennerblunsom_emnlp13)]
  * Nal Kalchbrenner and Phil Blunsom, *Recurrent Continuous Translation Models*, EMNLP 2013
* Univ. Montreal
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014 [[Paper](http://arxiv.org/pdf/1406.1078)]
  * Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio, *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*, SSST-8 2014 [[Paper](http://www.aclweb.org/anthology/W14-4012)]
  * Jean Pouget-Abadie, Dzmitry Bahdanau, Bart van Merrienboer, Kyunghyun Cho, and Yoshua Bengio, *Overcoming the Curse of Sentence Length for Neural Machine Translation using Automatic Segmentation*, SSST-8 2014
  * Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, arXiv:1409.0473 / ICLR 2015 [[Paper](http://arxiv.org/pdf/1409.0473)]
  * Sebastian Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio, *On using very large target vocabulary for neural machine translation*, arXiv:1412.2007 / ACL 2015 [[Paper](http://arxiv.org/pdf/1412.2007.pdf)]
* Univ. Montreal + Middle East Tech. Univ. + Univ. Maine [[Paper](http://arxiv.org/pdf/1503.03535.pdf)]
  * Caglar Gulcehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Loic Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *On Using Monolingual Corpora in Neural Machine Translation*, arXiv:1503.03535
* Google [[Paper](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)]
  * Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, *Sequence to Sequence Learning with Neural Networks*, arXiv:1409.3215 / NIPS 2014
* Google + NYU [[Paper](http://arxiv.org/pdf/1410.8206)]
  * Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, and Wojciech Zaremba, *Addressing the Rare Word Problem in Neural Machine Transltaion*, arXiv:1410.8206 / ACL 2015
* ICT + Huawei [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
  * Fandong Meng, Zhengdong Lu, Zhaopeng Tu, Hang Li, and Qun Liu, *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442
* Stanford [[Paper](http://arxiv.org/pdf/1508.04025.pdf)]
  * Minh-Thang Luong, Hieu Pham, and Christopher D. Manning, *Effective Approaches to Attention-based Neural Machine Translation*, arXiv:1508.04025
* Middle East Tech. Univ. + NYU + Univ. Montreal [[Paper](http://arxiv.org/pdf/1601.01073.pdf)]
  * Orhan Firat, Kyunghyun Cho, and Yoshua Bengio, *Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism*, arXiv:1601.01073

#### Conversation Modeling
* Lifeng Shang, Zhengdong Lu, and Hang Li, *Neural Responding Machine for Short-Text Conversation*, arXiv:1503.02364 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.02364)]
* Oriol Vinyals and Quoc V. Le, *A Neural Conversational Model*, arXiv:1506.05869 [[Paper](http://arxiv.org/pdf/1506.05869)]
* Ryan Lowe, Nissan Pow, Iulian V. Serban, and Joelle Pineau, *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*, arXiv:1506.08909 [[Paper](http://arxiv.org/pdf/1506.08909)]
* Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, and Jason Weston, *Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems*, arXiv:1511.06931 [[Paper](http://arxiv.org/pdf/1511.06931)]
* Jason Weston, *Dialog-based Language Learning*, arXiv:1604.06045, [[Paper](http://arxiv.org/pdf/1604.06045)]
* Antoine Bordes and Jason Weston, *Learning End-to-End Goal-Oriented Dialog*, arXiv:1605.07683 [[Paper](http://arxiv.org/pdf/1605.07683)]

#### Question Answering
* FAIR
  * Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, and Alexander M. Rush, *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*, arXiv:1502.05698 [[Web](https://research.facebook.com/researchers/1543934539189348)] [[Paper](http://arxiv.org/pdf/1502.05698.pdf)]
  * Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston, *Simple Question answering with Memory Networks*, arXiv:1506.02075 [[Paper](http://arxiv.org/abs/1506.02075)]
  * Felix Hill, Antoine Bordes, Sumit Chopra, Jason Weston, "The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations", ICLR 2016 [[Paper](http://arxiv.org/abs/1511.02301)]
* DeepMind + Oxford [[Paper](http://arxiv.org/pdf/1506.03340.pdf)]
  * Karl M. Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom, *Teaching Machines to Read and Comprehend*, arXiv:1506.03340 / NIPS 2015
* MetaMind [[Paper](http://arxiv.org/pdf/1506.07285.pdf)]
  * Ankit Kumar, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Mohit Iyyer, Ishaan Gulrajani, and Richard Socher, *Ask Me Anything: Dynamic Memory Networks for Natural Language Processing*, arXiv:1506.07285

### Computer Vision

#### Object Recognition
* Pedro Pinheiro and Ronan Collobert, *Recurrent Convolutional Neural Networks for Scene Labeling*, ICML 2014 [[Paper](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf)]
* Ming Liang and Xiaolin Hu, *Recurrent Convolutional Neural Network for Object Recognition*, CVPR 2015 [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)]
* Wonmin Byeon, Thomas Breuel, Federico Raue1, and Marcus Liwicki1, *Scene Labeling with LSTM Recurrent Neural Networks*, CVPR 2015 [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)]
* Mircea Serban Pavel, Hannes Schulz, and Sven Behnke, *Recurrent Convolutional Neural Networks for Object-Class Segmentation of RGB-D Video*, IJCNN 2015 [[Paper](http://www.ais.uni-bonn.de/papers/IJCNN_2015_Pavel.pdf)]
* Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip H. S. Torr, *Conditional Random Fields as Recurrent Neural Networks*, arXiv:1502.03240 [[Paper](http://arxiv.org/pdf/1502.03240)]
* Xiaodan Liang, Xiaohui Shen, Donglai Xiang, Jiashi Feng, Liang Lin, and Shuicheng Yan, *Semantic Object Parsing with Local-Global Long Short-Term Memory*, arXiv:1511.04510 [[Paper](http://arxiv.org/pdf/1511.04510.pdf)]
* Sean Bell, C. Lawrence Zitnick, Kavita Bala, and Ross Girshick, *Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks*, arXiv:1512.04143 / ICCV 2015 workshop [[Paper](http://arxiv.org/pdf/1512.04143)]

#### Visual Tracking
* Quan Gan, Qipeng Guo, Zheng Zhang, and Kyunghyun Cho, *First Step toward Model-Free, Anonymous Object Tracking with Recurrent Neural Networks*, arXiv:1511.06425 [[Paper](http://arxiv.org/pdf/1511.06425)]


#### Image Generation
* Karol Gregor, Ivo Danihelka, Alex Graves, Danilo J. Rezende, and Daan Wierstra, *DRAW: A Recurrent Neural Network for Image Generation,* ICML 2015 [[Paper](http://arxiv.org/pdf/1502.04623)]
* Angeliki Lazaridou, Dat T. Nguyen, R. Bernardi, and M. Baroni, *Unveiling the Dreams of Word Embeddings: Towards Language-Driven Image Generation,* arXiv:1506.03500 [[Paper](http://arxiv.org/pdf/1506.03500)]
* Lucas Theis and Matthias Bethge, *Generative Image Modeling Using Spatial LSTMs,* arXiv:1506.03478 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.03478)]
* Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu, *Pixel Recurrent Neural Networks,* arXiv:1601.06759 [[Paper](http://arxiv.org/abs/1601.06759)]

#### Video Analysis

* Univ. Toronto [[paper](http://arxiv.org/abs/1502.04681)]
  * Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, *Unsupervised Learning of Video Representations using LSTMs*, arXiv:1502.04681 / ICML 2015
* Univ. Cambridge [[paper](http://arxiv.org/abs/1511.06309)]
  * Viorica Patraucean, Ankur Handa, Roberto Cipolla, *Spatio-temporal video autoencoder with differentiable memory*, arXiv:1511.06309



### Multimodal (CV + NLP)

#### Image Captioning
* UCLA + Baidu [[Web](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html)] [[Paper-arXiv1](http://arxiv.org/pdf/1410.1090)], [[Paper-arXiv2](http://arxiv.org/pdf/1412.6632)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, and Alan L. Yuille, *Explain Images with Multimodal Recurrent Neural Networks*, arXiv:1410.1090
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)*, arXiv:1412.6632 / ICLR 2015
* Univ. Toronto [[Paper](http://arxiv.org/pdf/1411.2539)] [[Web demo](http://deeplearning.cs.toronto.edu/i2t)]
  * Ryan Kiros, Ruslan Salakhutdinov, and Richard S. Zemel, *Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models*, arXiv:1411.2539 / TACL 2015
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/pdf/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* Google [[Paper](http://arxiv.org/pdf/1411.4555)]
  * Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan, *Show and Tell: A Neural Image Caption Generator*, arXiv:1411.4555 / CVPR 2015
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
  * Andrej Karpathy and Li Fei-Fei, *Deep Visual-Semantic Alignments for Generating Image Description*, CVPR 2015
* Microsoft [[Paper](http://arxiv.org/pdf/1411.4952)]
  * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollar, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, Lawrence Zitnick, and Geoffrey Zweig, *From Captions to Visual Concepts and Back*, arXiv:1411.4952 / CVPR 2015
* CMU + Microsoft [[Paper-arXiv](http://arxiv.org/pdf/1411.5654)], [[Paper-CVPR](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)]
  * Xinlei Chen, and C. Lawrence Zitnick, *Learning a Recurrent Visual Representation for Image Caption Generation*
  * Xinlei Chen, and C. Lawrence Zitnick, *Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation*, CVPR 2015
* Univ. Montreal + Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]
  * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio, *Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention*, arXiv:1502.03044 / ICML 2015
* Idiap + EPFL + Facebook [[Paper](http://arxiv.org/pdf/1502.03671)]
  * Remi Lebret, Pedro O. Pinheiro, and Ronan Collobert, *Phrase-based Image Captioning*, arXiv:1502.03671 / ICML 2015
* UCLA + Baidu [[Paper](http://arxiv.org/pdf/1504.06692)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images*, arXiv:1504.06692
* MS + Berkeley
  * Jacob Devlin, Saurabh Gupta, Ross Girshick, Margaret Mitchell, and C. Lawrence Zitnick, *Exploring Nearest Neighbor Approaches for Image Captioning*, arXiv:1505.04467 (Note: technically not RNN) [[Paper](http://arxiv.org/pdf/1505.04467.pdf)]
  * Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, and Margaret Mitchell, *Language Models for Image Captioning: The Quirks and What Works*, arXiv:1505.01809 [[Paper](http://arxiv.org/pdf/1505.01809.pdf)]
* Adelaide [[Paper](http://arxiv.org/pdf/1506.01144.pdf)]
  * Qi Wu, Chunhua Shen, Anton van den Hengel, Lingqiao Liu, and Anthony Dick, *Image Captioning with an Intermediate Attributes Layer*, arXiv:1506.01144
* Tilburg [[Paper](http://arxiv.org/pdf/1506.03694.pdf)]
  * Grzegorz Chrupala, Akos Kadar, and Afra Alishahi, *Learning language through pictures*, arXiv:1506.03694
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, and Yoshua Bengio, *Describing Multimedia Content using Attention-based Encoder-Decoder Networks*, arXiv:1507.01053
* Cornell [[Paper](http://arxiv.org/pdf/1508.02091.pdf)]
  * Jack Hessel, Nicolas Savva, and Michael J. Wilber, *Image Representations and New Domains in Neural Image Captioning*, arXiv:1508.02091


#### Video Captioning
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/pdf/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* UT Austin + UML + Berkeley [[Paper](http://arxiv.org/pdf/1412.4729)]
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, and Kate Saenko, *Translating Videos to Natural Language Using Deep Recurrent Neural Networks*, arXiv:1412.4729
* Microsoft [[Paper](http://arxiv.org/pdf/1505.01861)]
  * Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, and Yong Rui, *Joint Modeling Embedding and Translation to Bridge Video and Language*, arXiv:1505.01861
* UT Austin + Berkeley + UML [[Paper](http://arxiv.org/pdf/1505.00487)]
  * Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko, *Sequence to Sequence--Video to Text*, arXiv:1505.00487
* Univ. Montreal + Univ. Sherbrooke [[Paper](http://arxiv.org/pdf/1502.08029.pdf)]
  * Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, and Aaron Courville, *Describing Videos by Exploiting Temporal Structure*, arXiv:1502.08029
* MPI + Berkeley [[Paper](http://arxiv.org/pdf/1506.01698.pdf)]
  * Anna Rohrbach, Marcus Rohrbach, and Bernt Schiele, *The Long-Short Story of Movie Description*, arXiv:1506.01698
* Univ. Toronto + MIT [[Paper](http://arxiv.org/pdf/1506.06724.pdf)]
  * Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler, *Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books*, arXiv:1506.06724
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, and Yoshua Bengio, *Describing Multimedia Content using Attention-based Encoder-Decoder Networks*, arXiv:1507.01053
* Zhejiang Univ. + UTS [[Paper](http://arxiv.org/abs/1511.03476)]
  * Pingbo Pan, Zhongwen Xu, Yi Yang, Fei Wu, Yueting Zhuang, *Hierarchical Recurrent Neural Encoder for Video Representation with Application to Captioning*, arXiv:1511.03476
* Univ. Montreal + NYU + IBM [[Paper](http://arxiv.org/pdf/1511.04590.pdf)]
  * Li Yao, Nicolas Ballas, Kyunghyun Cho, John R. Smith, and Yoshua Bengio, *Empirical performance upper bounds for image and video captioning*, arXiv:1511.04590


#### Visual Question Answering

* Virginia Tech. + MSR [[Web](http://www.visualqa.org/)] [[Paper](http://arxiv.org/pdf/1505.00468)]
  * Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh, *VQA: Visual Question Answering*, arXiv:1505.00468 / CVPR 2015 SUNw:Scene Understanding workshop
* MPI + Berkeley [[Web](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/)] [[Paper](http://arxiv.org/pdf/1505.01121)]
  * Mateusz Malinowski, Marcus Rohrbach, and Mario Fritz, *Ask Your Neurons: A Neural-based Approach to Answering Questions about Images*, arXiv:1505.01121
* Univ. Toronto [[Paper](http://arxiv.org/pdf/1505.02074)] [[Dataset](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)]
  * Mengye Ren, Ryan Kiros, and Richard Zemel, *Exploring Models and Data for Image Question Answering*, arXiv:1505.02074 / ICML 2015 deep learning workshop
* Baidu + UCLA [[Paper](http://arxiv.org/pdf/1505.05612)] [[Dataset]()]
  * Hauyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, and Wei Xu, *Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering*, arXiv:1505.05612 / NIPS 2015
* SNU + NAVER [[Paper](http://arxiv.org/abs/1606.01455)]
  * Jin-Hwa Kim, Sang-Woo Lee, Dong-Hyun Kwak, Min-Oh Heo, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Multimodal Residual Learning for Visual QA*, arXiv:1606:01455
* UC Berkeley + Sony [[Paper](https://arxiv.org/pdf/1606.01847)]
  * Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach, *Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding*, arXiv:1606.01847
* Postech [[Paper](http://arxiv.org/pdf/1606.03647.pdf)]
  * Hyeonwoo Noh and Bohyung Han, *Training Recurrent Answering Units with Joint Loss Minimization for VQA*, arXiv:1606.03647
* SNU + NAVER [[Paper](http://arxiv.org/abs/1610.04325)]
  * Jin-Hwa Kim, Kyoung Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Hadamard Product for Low-rank Bilinear Pooling*, arXiv:1610.04325
* Video QA
  * CMU + UTS [[paper](http://arxiv.org/abs/1511.04670)]
    * Linchao Zhu, Zhongwen Xu, Yi Yang, Alexander G. Hauptmann, Uncovering Temporal Context for Video Question and Answering, arXiv:1511.04670
  * KIT + MIT + Univ. Toronto [[Paper](http://arxiv.org/abs/1512.02902)] [[Dataset](http://movieqa.cs.toronto.edu/home/)]
    * Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, Sanja Fidler, MovieQA: Understanding Stories in Movies through Question-Answering, arXiv:1512.02902


#### Turing Machines
*  A.Graves, G. Wayne, and I. Danihelka., *Neural Turing Machines,* arXiv preprint arXiv:1410.5401 [[Paper](http://arxiv.org/pdf/1410.5401)]
* Jason Weston, Sumit Chopra, Antoine Bordes, *Memory Networks,* arXiv:1410.3916 [[Paper](http://arxiv.org/pdf/1410.3916)]
* Armand Joulin and Tomas Mikolov, *Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets*, arXiv:1503.01007 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.01007)]
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus, *End-To-End Memory Networks*, arXiv:1503.08895 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.08895)]
* Wojciech Zaremba and Ilya Sutskever, *Reinforcement Learning Neural Turing Machines,* arXiv:1505.00521 [[Paper](http://arxiv.org/pdf/1505.00521)]
* Baolin Peng and Kaisheng Yao, *Recurrent Neural Networks with External Memory for Language Understanding*, arXiv:1506.00195 [[Paper](http://arxiv.org/pdf/1506.00195.pdf)]
* Fandong Meng, Zhengdong Lu, Zhaopeng Tu, Hang Li, and Qun Liu, *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442 [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
* Arvind Neelakantan, Quoc V. Le, and Ilya Sutskever, *Neural Programmer: Inducing Latent Programs with Gradient Descent*, arXiv:1511.04834 [[Paper](http://arxiv.org/pdf/1511.04834.pdf)]
* Scott Reed and Nando de Freitas, *Neural Programmer-Interpreters*, arXiv:1511.06279 [[Paper](http://arxiv.org/pdf/1511.06279.pdf)]
* Karol Kurach, Marcin Andrychowicz, and Ilya Sutskever, *Neural Random-Access Machines*, arXiv:1511.06392 [[Paper](http://arxiv.org/pdf/1511.06392.pdf)]
* Łukasz Kaiser and Ilya Sutskever, *Neural GPUs Learn Algorithms*, arXiv:1511.08228 [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]
* Ethan Caballero, *Skip-Thought Memory Networks*, arXiv:1511.6420 [[Paper](https://pdfs.semanticscholar.org/6b9f/0d695df0ce01d005eb5aa69386cb5fbac62a.pdf)]
* Wojciech Zaremba, Tomas Mikolov, Armand Joulin, and Rob Fergus, *Learning Simple Algorithms from Examples*, arXiv:1511.07275 [[Paper](http://arxiv.org/pdf/1511.07275.pdf)]

### Robotics

* Hongyuan Mei, Mohit Bansal, and Matthew R. Walter, *Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to Action Sequences*, arXiv:1506.04089 [[Paper](http://arxiv.org/pdf/1506.04089.pdf)]
* Marvin Zhang, Sergey Levine, Zoe McCarthy, Chelsea Finn, and Pieter Abbeel, *Policy Learning with Continuous Memory States for Partially Observed Robotic Control,* arXiv:1507.01273. [[Paper]](http://arxiv.org/pdf/1507.01273)

### Other
* Alex Graves, *Generating Sequences With Recurrent Neural Networks,* arXiv:1308.0850 [[Paper]](http://arxiv.org/abs/1308.0850)
* Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu, *Recurrent Models of Visual Attention*, NIPS 2014 / arXiv:1406.6247 [[Paper](http://arxiv.org/pdf/1406.6247.pdf)]
* Wojciech Zaremba and Ilya Sutskever, *Learning to Execute*, arXiv:1410.4615 [[Paper](http://arxiv.org/pdf/1410.4615.pdf)] [[Code](https://github.com/wojciechz/learning_to_execute)]
* Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer, *Scheduled Sampling for Sequence Prediction with
Recurrent Neural Networks*, arXiv:1506.03099 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.03099)]
* Bing Shuai, Zhen Zuo, Gang Wang, and Bing Wang, *DAG-Recurrent Neural Networks For Scene Labeling*, arXiv:1509.00552 [[Paper](http://arxiv.org/pdf/1509.00552)]
* Soren Kaae Sonderby, Casper Kaae Sonderby, Lars Maaloe, and Ole Winther, *Recurrent Spatial Transformer Networks*, arXiv:1509.05329 [[Paper](http://arxiv.org/pdf/1509.05329)]
* Cesar Laurent, Gabriel Pereyra, Philemon Brakel, Ying Zhang, and Yoshua Bengio, *Batch Normalized Recurrent Neural Networks*, arXiv:1510.01378 [[Paper](http://arxiv.org/pdf/1510.01378)]
* Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee, *Deeply-Recursive Convolutional Network for Image Super-Resolution*, arXiv:1511.04491 [[Paper]](http://arxiv.org/abs/1511.04491)
* Quan Gan, Qipeng Guo, Zheng Zhang, and Kyunghyun Cho, *First Step toward Model-Free, Anonymous Object Tracking with Recurrent Neural Networks*, arXiv:1511.06425 [[Paper](http://arxiv.org/pdf/1511.06425.pdf)]
* Francesco Visin, Kyle Kastner, Aaron Courville, Yoshua Bengio, Matteo Matteucci, and Kyunghyun Cho, *ReSeg: A Recurrent Neural Network for Object Segmentation*, arXiv:1511.07053 [[Paper](http://arxiv.org/pdf/1511.07053.pdf)]
* Juergen Schmidhuber, *On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models*, arXiv:1511.09249 [[Paper]](http://arxiv.org/pdf/1511.09249)

## Datasets
* Speech Recognition
  * [OpenSLR](http://www.openslr.org/resources.php) (Open Speech and Language Resources)
    * [LibriSpeech ASR corpus](http://www.openslr.org/12/)
  * [VoxForge](http://voxforge.org/home)
* Image Captioning
  * [Flickr 8k](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
  * [Flickr 30k](http://shannon.cs.illinois.edu/DenotationGraph/)
  * [Microsoft COCO](http://mscoco.org/home/)
* Question Answering
  * [The bAbI Project](http://fb.ai/babi) - Dataset for text understanding and reasoning, by Facebook AI Research. Contains:
    * The (20) QA bAbI tasks - [[Paper](http://arxiv.org/abs/1502.05698)]
    * The (6) dialog bAbI tasks - [[Paper](http://arxiv.org/abs/1605.07683)]
    * The Children's Book Test - [[Paper](http://arxiv.org/abs/1511.02301)]
    * The Movie Dialog dataset - [[Paper](http://arxiv.org/abs/1511.06931)]
    * The MovieQA dataset - [[Data](http://www.thespermwhale.com/jaseweston/babi/movie_dialog_dataset.tgz)]
    * The Dialog-based Language Learning dataset - [[Paper](http://arxiv.org/abs/1604.06045)]
    * The SimpleQuestions dataset - [[Paper](http://arxiv.org/abs/1506.02075)]
  * [SQuAD](https://stanford-qa.com/) - Stanford Question Answering Dataset :  [[Paper](http://arxiv.org/pdf/1606.05250)]
* Image Question Answering
  * [DAQUAR](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/) - built upon [NYU Depth v2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) by N. Silberman et al.
  * [VQA](http://www.visualqa.org/) - based on [MSCOCO](http://mscoco.org/) images
  * [Image QA](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/) - based on MSCOCO images
  * [Multilingual Image QA](http://idl.baidu.com/FM-IQA.html) - built from scratch by Baidu - in Chinese, with English translation
* Action Recognition
  * [THUMOS](http://www.thumos.info/home.html) : Large-scale action recognition dataset
  * [MultiTHUMOS](http://ai.stanford.edu/~syyeung/resources/multithumos.zip) : Extension of THUMOS '14 action detection dataset with dense multilabele annotation

## Blogs
* [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) in [Colah's blog](http://colah.github.io/)
* [WildML](http://www.wildml.com/) blog's RNN tutorial [[Part1](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)], [[Part2](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)], [[Part3](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)], [[Part4](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)]
* [RNNs in Tensorflow, a Practical Guide and Undocumented Features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [Optimizing RNN Performance](https://svail.github.io/) from Baidu's Silicon Valley AI Lab.
* [Character Level Language modelling using RNN](http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139) by Yoav Goldberg
* [Implement an RNN in Python](http://peterroelants.github.io/posts/rnn_implementation_part01/).
* [LSTM Backpropogation](http://arunmallya.github.io/writeups/nn/lstm/index.html#/)
* [Introduction to Recurrent Networks in TensorFlow](https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/) by Danijar Hafner
* [Variable Sequence Lengths in TensorFlow](https://danijar.com/variable-sequence-lengths-in-tensorflow/) by Danijar Hafner
* [Written Memories: Understanding, Deriving and Extending the LSTM](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html) by Silviu Pitis

## Online Demos
* Alex graves, hand-writing generation [[link](http://www.cs.toronto.edu/~graves/handwriting.html)]
* Ink Poster: Handwritten post-it notes [[link](http://www.inkposter.com/?)]
* LSTMVis: Visual Analysis for Recurrent Neural Networks [[link](http://lstm.seas.harvard.edu/)]
