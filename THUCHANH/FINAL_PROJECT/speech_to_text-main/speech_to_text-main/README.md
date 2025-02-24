# speech_to_text
This Project includes implementation of Encoder-Decoder model and implementation of RNN  Encoder-Decoder (encoding-decoding) is a very common model framework in deep learning. For example, auto-encoding of unsupervised algorithms is designed and trained with the structure of encoding-decoding; Is the encoding-decoding framework of CNN-RNN; for example, the neural network machine translation NMT model is often the LSTM-LSTM encoding-decoding framework.  Therefore, to be precise, Encoder-Decoder is not a specific model, but a kind of framework. Encoder and Decoder parts can be any text, voice, image, video data, and models can use CNN, RNN, BiRNN, LSTM, GRU, etc. So based on Encoder-Decoder, we can design a variety of application algorithms.  Here we have made use of voice  One of the most significant features of the Encoder-Decoder framework is that it is an End-to-End learning algorithm. Such models are often used in machine translation, such as translating one language to another . Such a model is also called Sequence to Sequence learning. The so-called encoding is to convert the input sequence into a fixed-length vector and decoding is to convert the previously generated fixed vector into an output sequence.
Python libraries used for this project -
- librosa            - for audio processing
- os                 - for operating system files
- numpy              - for numerical analysis
- matplotlib         - for data visualization
- scipy              - for wavfile handling
- speech_recognition - for translating speech to text
- pydub              - working with wav files
- tensorflow         - machine learning library
- keras              - neural network library(runs on top of tensorflow)
