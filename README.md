# Project

Audio Project bits
--------------------
Current Plan
-----------
Steps:


1.   Take a collection of songs as input by requesting audio data from Spotify's API 
2.   Use librosa to convert audio data into spectrogram images 
3. Classify their genres using our CNN model trained on GTZAN
4. Take each song's softmax output to obtain the calculated probability of each genre
5. Interpret each soft-max output as position vectors in a multi-dimensional vector space, from 0-1 in each genre.
6. Order the track listing in such a way to  approximate a minimal S(X), the sum of each distance (some chosen distance/cost function d(x,y)), between consecutive songs in the vector space.
7.Output a track listing which gives our attempt at ordering songs into the most natural progression between genres


Want to solve the problem of finding the shortest route through a low (< 20) number of points in a multi-dimensional space. 
We attempt this using a range of algorithms (Branch & Bound search algorithm, genetic algorithm using mlrose) and discuss their efficiency:

### Travelling Salesperson Search/Dynamic Programming Algorithms
-----

https://arxiv.org/pdf/1805.04131.pdf

https://coral.ise.lehigh.edu/wiki/doku.php/travelling_salesman_problem

https://en.wikipedia.org/wiki/Genetic_algorithm

https://towardsdatascience.com/solving-travelling-salesperson-problems-with-python-5de7e883d847
Project Proposal shared google doc https://docs.google.com/document/d/1rYBPLHOS148kl6nmSgqdELuqsLe_-0z8uCnpsoIHHBw/edit?usp=sharing
...

Potential Applications
-------------
#### A map of all of spotify's data - pretty cool (type in your fave artist and it'll show you their specific genre and all the most similar artists to them) http://everynoise.com/#otherthings

#### Use Generative Adversarial Networks (GAN's) to create new raw audio
- cool video: Using GAN's to turn jump-up DnB into liquid -https://www.youtube.com/watch?v=KMsFgts23x4&ab_channel=AIPursuit-YourDailyAIDigest
- nice simple blogpost on gans https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
- GANSynth - make one instrument sound like another: https://magenta.tensorflow.org/gansynth
- Wavenet - generates raw audio, pretty demanding computationally : https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/

Useful Links for each section
-----------------------------------------------------------------------------------


Classifying Audio papers(from Ben)
--------------
-  https://www.kdnuggets.com/2017/12/audio-classifier-deep-neural-networks.html
- https://arxiv.org/pdf/1901.04555.pdf


TensorFLow (Python Library for Deep Learning/Neural Nets)
---------------
- Simple Intro tutorial walkthroughs : https://www.tensorflow.org/tutorials
- Creating spectograms and classifying audio on TensorFlow : https://www.tensorflow.org/tutorials/audio/simple_audio
- Using DCGAN's on TensorFLow : https://www.tensorflow.org/tutorials/generative/dcgan

Udemy Course link:
----------------------
https://www.udemy.com/course/artificial-intelligence-mastercalss/


[Archived] GAN's
---------------
- nice simple blogpost https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
- Simple google intro https://developers.google.com/machine-learning/gan
- Audio GAN: https://magenta.tensorflow.org/gansynth

[Archived] Spotify (Klio) dataset 
-------------
- Quickstart info https://docs.klio.io/en/latest/quickstart/index.html
- Klio Audio Separation and Spectogram Example : https://docs.klio.io/en/latest/userguide/examples/audio_spec.html
- Klio Documentation : https://klio.readthedocs.io/en/latest/userguide/index.html
- Klio Github : https://github.com/spotify/klio
- Lecture on using Apache Beam in Python : https://www.youtube.com/watch?v=I1JUtoDHFcg&ab_channel=PyConSG
- Google Pub/Sub tutorial (completely dull, but needed to request datafrom spotify 'pipelines?') : https://console.cloud.google.com/cloudpubsub/topic/list?tutorial=pubsub_quickstart&project=api-project-958593700903
- Apache Beam Walkthrough example : https://beam.apache.org/get-started/wordcount-example/
- google account for data storage login : email: ai.music.generation@gmail.com password: McConville
- Docker & WSL 2 on Windows Download required for Klio https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers
- Docker tutorial for beginners : https://docker-curriculum.com/

CNN and related Blogs:
-------------
https://www.kaggle.com/andradaolteanu/work-w-audio-data-visualise-classify-recommend/data#Machine-Learning-Classification
https://www.kaggle.com/vicky1999/music-genre-classification/notebook#Dataset
https://towardsdatascience.com/music-genre-recognition-using-convolutional-neural-networks-cnn-part-1-212c6b93da76
