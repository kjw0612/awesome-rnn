# Awesome Recurrent Neural Networks

A curated list of resources dedicated to recurrent neural networks (closely related to *deep learning*).

Maintainers - [Jiwon Kim](http://github.com/kjw0612), [Myungsub Choi](http://github.com/myungsub)

We have pages for other topics: [awesome-deep-vision](http://jiwonkim.org/awesome-deep-vision/), [awesome-random-forest](http://jiwonkim.org/awesome-random-forest/)


## Contributing
Please feel free to [pull requests](https://github.com/kjw0612/awesome-deep-vision/pulls), email Myungsub Choi (myungsub91@gmail.com) or join our chats to add links.

[![Join the chat at https://gitter.im/kjw0612/awesome-rnn](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/kjw0612/awesome-rnn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Sharing
+ [Share on Twitter](http://twitter.com/home?status=http://jiwonkim.org/awesome-rnn%0AResources for Recurrent Neural Networks)
+ [Share on Facebook](http://www.facebook.com/sharer/sharer.php?u=https://jiwonkim.org/awesome-rnn)
+ [Share on Google Plus](http://plus.google.com/share?url=https://jiwonkim.org/awesome-rnn)
+ [Share on LinkedIn](http://www.linkedin.com/shareArticle?mini=true&url=https://jiwonkim.org/awesome-rnn&title=Awesome%20Recurrent%20Neural&Networks&summary=&source=)

## Table of Contents

 - [Codes](#codes)
 - [Theory](#theory)
   - [Lectures](#lectures)
   - [Books / Thesis](#books--thesis)
   - [Network Variants](#network-variants)
   - [Surveys](#surveys)
 - [Applications](#applications)
   - [Language Modeling](#language-modeling)
   - [Speech Recognition](#speech-recognition)
   - [Machine Translation](#machine-translation)
   - [Conversation Modeling](#conversation-modeling)
   - [Image Captioning](#image-captioning)
   - [Video Captioning](#video-captioning)
   - [Question Answering](#question-answering)
   - [Image Generation](#image-generation)
   - [Turing Machines](#turing-machines)
   - [Robotics](#robotics)
 - [Datasets](#datasets)

## Codes
* [Theano](http://deeplearning.net/software/theano/) - Python
  * Simple IPython [tutorial on Theano](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb)
  * [Deep Learning Tutorials](http://www.deeplearning.net/tutorial/)
    * [RNN for semantic parsing of speech](http://www.deeplearning.net/tutorial/rnnslu.html#rnnslu)
    * [LSTM network for sentiment analysis](http://www.deeplearning.net/tutorial/lstm.html#lstm)
  * [Pylearn2](http://deeplearning.net/software/pylearn2/) : Library that wraps a lot of models and training algorithms in deep learning
  * [Blocks](https://github.com/mila-udem/blocks) : modular framework that enables building neural network models
  * [Keras](http://keras.io/) : Theano-based deep learning library similar to Torch, but in Python
  * [Lasagne](https://github.com/Lasagne/Lasagne) : Lightweight library to build and train neural networks in Theano
  * [theano-rnn](https://github.com/gwtaylor/theano-rnn) by Graham Taylor
  * [Passage](https://github.com/IndicoDataSolutions/Passage) : Library for text analysis with RNNs
* [Caffe](https://github.com/BVLC/caffe) - C++ with MATLAB/Python wrappers
  * [LRCN](http://jeffdonahue.com/lrcn/) by Jeff Donahue
* [Torch](http://torch.ch/) - Lua
  * [char-rnn](https://github.com/karpathy/char-rnn) by Andrej Karpathy : multi-layer RNN/LSTM/GRU for training/sampling from character-level language models
  * [LSTM](https://github.com/wojzaremba/lstm) by Wojciech Zaremba : Long Short Term Memory Units to train a language model on word level Penn Tree Bank dataset
  * [Oxford](https://github.com/oxford-cs-ml-2015) by Nando de Freitas : Oxford Computer Science - Machine Learning 2015 Practicals
  * [rnn](https://github.com/Element-Research/rnn) by Nicholas Leonard : general library for implementing RNN, LSTM, BRNN and BLSTM (highly unit tested).
* Etc.
  * [Chainer](http://chainer.org/) : new, flexible deep learning library in Python
  * [RNNLIB](http://sourceforge.net/p/rnnl/wiki/Home/) by Alex Graves : C++ based LSTM library
  * [RNNLM](http://rnnlm.org/) by Tomas Mikolov : C++ based simple code
  * [neuraltalk](https://github.com/karpathy/neuraltalk) by Andrej Karpathy : numpy-based RNN/LSTM implementation
  * [gist](https://gist.github.com/karpathy/587454dc0146a6ae21fc) by Andrej Karpathy : raw numpy code that implements an efficient batched LSTM
 
## Theory
### Lectures
* Stanford NLP ([CS224d](http://cs224d.stanford.edu/index.html)) by Richard Socher
  * [Lecture Note 3](http://cs224d.stanford.edu/lecture_notes/LectureNotes3.pdf) : neural network basics
  * [Lecture Note 4](http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf) : RNN language models, bi-directional RNN, GRU, LSTM
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

### Network Variants
* Bi-directional RNN [[Paper](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]
  * Mike Schuster and Kuldip K. Paliwal, *Bidirectional Recurrent Neural Networks*, Trans. on Signal Processing 1997
* LSTM [[Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)]
  * Sepp Hochreiter and Jurgen Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997
* Multi-dimensional RNN [[Paper](http://arxiv.org/pdf/0705.2011v1.pdf)]
  * Alex Graves, Santiago Fernandez, and Jurgen Schmidhuber, *Multi-Dimensional Recurrent Neural Networks*, ICANN 2007
* GRU (Gated Recurrent Unit) [[Paper](http://arxiv.org/pdf/1406.1078v3.pdf)]
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014
* GFRNN [[Paper-arXiv](http://arxiv.org/abs/1502.02367)] [[Paper-ICML](http://jmlr.org/proceedings/papers/v37/chung15.pdf)] [[Supplementary](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)]
  * Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio, *Gated Feedback Recurrent Neural Networks*, arXiv:1502.02367 / ICML 2015
* Tree-Structured LSTM [[Paper](http://arxiv.org/abs/1503.00075)]
  * Kai Sheng Tai, Richard Socher, and Christopher D. Manning, arXiv:1503.00075 / ACL 2015
* Grid LSTM [[Paper](http://arxiv.org/abs/1507.01526)]
  * Nal Kalchbrenner, Ivo Danihelka, and Alex Graves, *Grid Long Short-Term Memory*, arXiv:1507.01526

### Surveys
* Klaus Greff, Rupesh Kumar Srivastava, Jan Koutnik, Bas R. Steunebrink, Jurgen Schmidhuber, [LSTM: A Search Space Odyssey](http://arxiv.org/abs/1503.04069), arXiv:1503.04069
* Zachary C. Lipton, [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/abs/1506.00019), arXiv:1506.00019
* Andrej Karpathy, Justin Johnson, Li Fei-Fei, [Visualizing and Understanding Recurrent Networks](http://arxiv.org/abs/1506.02078), arXiv:1506.02078
* Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever, [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf), ICML, 2015.

## Applications

### Language Modeling
* Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Recurrent Neural Network based Language Model*, Interspeech 2010 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)]
* Tomas Mikolov, Stefan Kombrink, Lukas Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur, *Extensions of Recurrent Neural Network Language Model*, ICASSP 2011 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)]
* Stefan Kombrink, Tomas Mikolov, Martin Karafiat, Lukas Burget, *Recurrent Neural Network based Language Modeling in Meeting Recognition*, Interspeech 2011 [[Paper](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)]
* Jiwei Li, Minh-Thang Luong, and Dan Jurafsky, *A Hierarchical Neural Autoencoder for Paragraphs and Documents*, ACL 2015 [[Paper](http://arxiv.org/abs/1506.01057)], [[Code](https://github.com/jiweil/Hierarchical-Neural-Autoencoder)]

### Speech Recognition
* Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, and Brian Kingsbury, *Deep Neural Networks for Acoustic Modeling in Speech Recognition*, IEEE Signam Processing Magazine 2012 [[Paper](http://cs224d.stanford.edu/papers/maas_paper.pdf)]
* Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton, *Speech Recognition with Deep Recurrent Neural Networks*, arXiv:1303.5778 / ICASSP 2013 [[Paper](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)]
* Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, *Attention-Based Models for Speech Recognition*, arXiv:1506.07503 [[Paper](http://arxiv.org/abs/1506.07503)]

### Machine Translation
* Univ. Montreal [[Paper](http://arxiv.org/abs/1406.1078)]
  * Kyunghyun Cho, Bart van Berrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio, *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014
* Google [[Paper](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)]
  * Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, *Sequence to Sequence Learning with Neural Networks*, arXiv:1409.3215 / NIPS 2014
* Univ. Montreal [[Paper](http://arxiv.org/abs/1409.0473)]
  * Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, arXiv:1409.0473 / ICLR 2015
* Google + NYU [[Paper](http://arxiv.org/abs/1410.8206)]
  * Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, and Wojciech Zaremba, *Addressing the Rare Word Problem in Neural Machine Transltaion*, ACL 2015

### Conversation Modeling
* Lifeng Shang, Zhengdong Lu, and Hang Li, *Neural Responding Machine for Short-Text Conversation*, arXiv:1503.02364 / ACL 2015 [[Paper](http://arxiv.org/abs/1503.02364)]
* Oriol Vinyals and Quoc V. Le, *A Neural Conversational Model*, arXiv:1506.05869 [[Paper](http://arxiv.org/abs/1506.05869)]
* Ryan Lowe, Nissan Pow, Iulian V. Serban, and Joelle Pineau, *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*, arXiv:1506.08909 [[Paper](http://arxiv.org/abs/1506.08909)]

### Image Captioning
* UCLA + Baidu [[Web](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html)] [[Paper-arXiv1](http://arxiv.org/abs/1410.1090)], [[Paper-arXiv2](http://arxiv.org/abs/1412.6632)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, and Alan L. Yuille, *Explain Images with Multimodal Recurrent Neural Networks*, arXiv:1410.1090
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)*, arXiv:1412.6632 / ICLR 2015
* Univ. Toronto [[Paper](http://arxiv.org/abs/1411.2539)] [[Web demo](http://deeplearning.cs.toronto.edu/i2t)]
  * Ryan Kiros, Ruslan Salakhutdinov, and Richard S. Zemel, *Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models*, arXiv:1411.2539 / TACL 2015
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/abs/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* Google [[Paper](http://arxiv.org/abs/1411.4555)]
  * Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan, *Show and Tell: A Neural Image Caption Generator*, arXiv:1411.4555 / CVPR 2015
* Microsoft [[Paper](http://arxiv.org/abs/1411.4952)]
  * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollar, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, Lawrence Zitnick, and Geoffrey Zweig, *From Captions to Visual Concepts and Back*, arXiv:1411.4952 / CVPR 2015
* Microsoft [[Paper-arXiv](http://arxiv.org/abs/1411.5654)], [[Paper-CVPR](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)]
  * Xinlei Chen, and C. Lawrence Zitnick, *Learning a Recurrent Visual Representation for Image Caption Generation*
  * Xinlei Chen, and C. Lawrence Zitnick, *Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation*, CVPR 2015
* Univ. Montreal + Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]
  * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio, *Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention*, arXiv:1502.03044 / ICML 2015
* Idiap + EPFL + Facebook [[Paper](http://arxiv.org/abs/1502.03671)]
  * Remi Lebret, Pedro O. Pinheiro, and Ronan Collobert, *Phrase-based Image Captioning*, arXiv:1502.03671 / ICML 2015
* UCLA + Baidu [[Paper](http://arxiv.org/abs/1504.06692)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images*, arXiv:1504.06692

### Video Captioning
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/abs/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* UT Austin + UML + Berkeley [[Paper](http://arxiv.org/abs/1412.4729)]
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, and Kate Saenko, *Translating Videos to Natural Language Using Deep Recurrent Neural Networks*, arXiv:1412.4729
* Microsoft [[Paper](http://arxiv.org/abs/1505.01861)]
  * Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, and Yong Rui, *Joint Modeling Embedding and Translation to Bridge Video and Language*, arXiv:1505.01861
* UT Austin + Berkeley + UML [[Paper](http://arxiv.org/abs/1505.00487)]
  * Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko, *Sequence to Sequence--Video to Text*, arXiv:1505.00487

### Question Answering
* Virginia Tech. + MSR [[Web](http://www.visualqa.org/)] [[Paper](http://arxiv.org/abs/1505.00468)]
  * Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh, *VQA: Visual Question Answering*, arXiv:1505.00468 / CVPR 2015 SUNw:Scene Understanding workshop
* MPI + Berkeley [[Web](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/)] [[Paper](http://arxiv.org/abs/1505.01121)]
  * Mateusz Malinowski, Marcus Rohrbach, and Mario Fritz, *Ask Your Neurons: A Neural-based Approach to Answering Questions about Images*, arXiv:1505.01121
* Univ. Toronto [[Paper](http://arxiv.org/abs/1505.02074)] [[Dataset](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)]
  * Mengye Ren, Ryan Kiros, and Richard Zemel, *Exploring Models and Data for Image Question Answering*, arXiv:1505.02074 / ICML 2015 deep learning workshop
* Baidu + UCLA [[Paper](http://arxiv.org/abs/1505.05612)] [[Dataset]()]
  * Hauyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, and Wei Xu, *Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering*, arXiv:1505.05612

### Image Generation
* Karol Gregor, Ivo Danihelka, Alex Graves, Danilo J. Rezende, and Daan Wierstra, *DRAW: A Recurrent Neural Network for Image Generation,* ICML 2015 [[Paper](http://arxiv.org/abs/1502.04623)]
* Angeliki Lazaridou, Dat T. Nguyen, R. Bernardi, and M. Baroni, *Unveiling the Dreams of Word Embeddings: Towards Language-Driven Image Generation,* arXiv:1506.03500 [[Paper](http://arxiv.org/abs/1506.03500)]
* Lucas Theis and Matthias Bethge, *Generative Image Modeling Using Spatial LSTMs,* arXiv:1506.03478 [[Paper](http://arxiv.org/abs/1506.03478)]

### Turing Machines
*  A.Graves, G. Wayne, and I. Danihelka., *Neural Turing Machines,* arXiv preprint arXiv:1410.5401 [[Paper](http://arxiv.org/abs/1410.5401)]
* Jason Weston, Sumit Chopra, Antoine Bordes, *Memory Networks,* arXiv:1410.3916 [[Paper](http://arxiv.org/abs/1410.3916)]
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus, *End-To-End Memory Networks*, arXiv:1503.08895 [[Paper](http://arxiv.org/abs/1503.08895)]
* Wojciech Zaremba and Ilya Sutskever, *Reinforcement Learning Neural Turing Machines,* arXiv:1505.00521 [[Paper](http://arxiv.org/abs/1505.00521)]

### Robotics
* Marvin Zhang, Sergey Levine, Zoe McCarthy, Chelsea Finn, Pieter Abbeel, *Policy Learning with Continuous Memory States for Partially Observed Robotic Control,* arXiv:1507.01273. [[Paper]](http://arxiv.org/abs/1507.01273)


## Datasets
* Speech Recognition
  * [OpenSLR](http://www.openslr.org/resources.php) (Open Speech and Language Resources)
    * [LibriSpeech ASR corpus](http://www.openslr.org/12/)
  * [VoxForge](http://voxforge.org/home)
* Image Captioning
  * [Flickr 8k](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
  * [Flickr 30k](http://shannon.cs.illinois.edu/DenotationGraph/)
  * [Microsoft COCO](http://mscoco.org/home/)
* Image Question Answering - all based on MS COCO images
  * [VQA](http://www.visualqa.org/)
  * [Image QA](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
  * [Multilingual Image QA] : in Chinese, with English translation

