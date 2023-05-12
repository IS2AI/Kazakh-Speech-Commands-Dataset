# Kazakh-Speech-Commands-Dataset
The Kazakh Speech Commands Benchmark dataset and the details of training & testing of the model can be found in the [Keyword-MLP](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/tree/main/Keyword-MLP) directory. 

The source code of the followings will be uploaded soon:
- Synthetic Data Generation using Text-To-Speech Models
- Conversion of a PyTorch Model (.pth) to ONNX and TensorFlow Lite  

# Speech Corpus Scraping
In order to automatically extract speech commands from a large-scale speech corpus, we used [Vosk Speech Recognition Toolkit](https://github.com/alphacep/vosk-api/tree/master). The example code is given in [```speech_corpus_scraping.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/speech_corpus_scraping.ipynb) notebook. 

## Preprint 
[Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need](https://www.techrxiv.org/articles/preprint/Speech_Command_Recognition_Text-to-Speech_and_Speech_Corpus_Scraping_Are_All_You_Need/22717657)

## Citation
```
@article{Kuzdeuov2023,
author = "Askat Kuzdeuov and Shakhizat Nurgaliyev and Diana Turmakhan and Nurkhan Laiyk and Huseyin Atakan Varol",
title = "{Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need}",
year = "2023",
month = "5",
url = "https://www.techrxiv.org/articles/preprint/Speech_Command_Recognition_Text-to-Speech_and_Speech_Corpus_Scraping_Are_All_You_Need/22717657",
doi = "10.36227/techrxiv.22717657.v1"
}
```
