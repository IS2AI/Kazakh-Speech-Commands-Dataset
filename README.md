# Kazakh-Speech-Commands-Dataset

## Preprint 
[Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need](https://www.techrxiv.org/articles/preprint/Speech_Command_Recognition_Text-to-Speech_and_Speech_Corpus_Scraping_Are_All_You_Need/22717657)

## Speech corpus scraping
To automatically extract speech commands from a large-scale speech corpus, we used [Vosk Speech Recognition Toolkit](https://github.com/alphacep/vosk-api/tree/master). The example code is given in [```speech_corpus_scraping.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/speech_corpus_scraping.ipynb) notebook. 

## Synthetic speech commands generation
In this project, we used [Piper](https://github.com/rhasspy/piper) to generate synthetic speech commands. Piper is a fast, local neural text to speech system. It provides five voices for the Kazakh language. The list of available models for other languages can be found [here](https://github.com/rhasspy/piper/releases/tag/v0.0.2) and the corresponding demos are given [here](https://rhasspy.github.io/piper-samples/). To generate synthetic speech commands for Kazakh, download and unzip the model from [Google Drive](https://drive.google.com/file/d/1vfSIK_xSh-GY2GeW1_JGcrAba8mdZxpD/view?usp=share_link). Then, open the [```synthetic_data_generation.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/synthetic_data_generation.ipynb) notebook, update the path to the model, and run all cells. 

## Data augmentation 
To increase the dataset size further, you can apply audio augmentation methods to the synthetic dataset and also to the speech corpus scraped dataset. The details can be found in the [```data_augmentation.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/data_augmentation.ipynb) notebook.

## Model training, validation, and testing
The details of training, validation, and testing of the model can be found in the [Keyword-MLP](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/tree/main/Keyword-MLP) directory. 

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
