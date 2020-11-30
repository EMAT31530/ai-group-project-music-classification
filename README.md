# Project

Audio Project bits
--------------------
Potential Current Plan
-----------
1) Call and extract data from spotify (Klio) data
2) Learn how to create Spectograms from spotify audio data
3) Create dataset of spectograms and useful spotify labels/attributes (artists, genre, year, liveliness, acoustic etc) 
4) Create AI Model (TensorFLow?) to train on spectograms with labels from spotify
5) Use this trained model to classify new audio from separate spotify song data
6) What can we now do with this? 
...

Applications we could use the data for
-------------
- some guy has done a mapping of all the spotify data - pretty cool (type in your fave artist and it'll show you their specific genre and all the most similar artists to them) http://everynoise.com/#otherthings

- Use Generative Adversarial Networks (GAN's) to create new raw audio
-- cool video: Using GAN's to turn jump-up DnB into liquid -https://www.youtube.com/watch?v=KMsFgts23x4&ab_channel=AIPursuit-YourDailyAIDigest
-- nice simple blogpost on gans https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
-- GANSynth - make one instrument sound like another: https://magenta.tensorflow.org/gansynth
-- Wavenet - generates raw audio, pretty demanding computationally : https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/

Useful Links for each section
-----------------------------------------------------------------------------------

Spotify (Klio) dataset 
-------------
- Quickstart info https://docs.klio.io/en/latest/quickstart/index.html
- Klio Audio Separation and Spectogram Example : https://docs.klio.io/en/latest/userguide/examples/audio_spec.html
- Klio Documentation : https://klio.readthedocs.io/en/latest/userguide/index.html
- Klio Github : https://github.com/spotify/klio
- Lecture on using Apache Beam in Python : https://www.youtube.com/watch?v=I1JUtoDHFcg&ab_channel=PyConSG
-- Google Pub/Sub tutorial (boring as shit needed to request datafrom spotify 'pipelines?') : https://console.cloud.google.com/cloudpubsub/topic/list?tutorial=pubsub_quickstart&project=api-project-958593700903
-- Apache Beam Walkthrough example : https://beam.apache.org/get-started/wordcount-example/
google account for data storage login : email: ai.music.generation@gmail.com password: McConville
-- Docker & WSL 2 Download required for Klio https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers


Classifying Audio papers(from Ben)
--------------
-  https://www.kdnuggets.com/2017/12/audio-classifier-deep-neural-networks.html
- https://arxiv.org/pdf/1901.04555.pdf


TensorFLow (Python Library for Deep Learning/Neural Nets)
---------------
- Simple Intro tutorial walkthroughs : https://www.tensorflow.org/tutorials
- Creating spectograms and classifying audio on TensorFlow : https://www.tensorflow.org/tutorials/audio/simple_audio
- Using DCGAN's on TensorFLow : https://www.tensorflow.org/tutorials/generative/dcgan

GAN Stuff 
---------------
- nice simple blogpost on gans https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
- Simple google intro to GAN's https://developers.google.com/machine-learning/gan
- Audio GAN: https://magenta.tensorflow.org/gansynth
