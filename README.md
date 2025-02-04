# Kazakh-Speech-Commands-Dataset

## Preprint 
[Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need](https://www.techrxiv.org/articles/preprint/Speech_Command_Recognition_Text-to-Speech_and_Speech_Corpus_Scraping_Are_All_You_Need/22717657)

## Paper on IEEE
[Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need](https://ieeexplore.ieee.org/document/10601292)

## Presentation on the 3rd International Conference on Robotics, Automation, and Artificial Intelligence (RAAI 2023) 
[Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need](https://docs.google.com/presentation/d/1oybWIY0SGu0y97eHZ393TLAyr-Dwy80S9NDIFVCVAnI/edit?usp=sharing)


## Synthetic speech commands generation
In this project, we used [Piper](https://github.com/rhasspy/piper) to generate synthetic speech commands. Piper is a fast, local neural text to speech system. It provides five voices for the Kazakh language. The list of available models for other languages can be found [here](https://github.com/rhasspy/piper/releases/tag/v0.0.2) and the corresponding demos are given [here](https://rhasspy.github.io/piper-samples/). To generate synthetic speech commands for Kazakh, download and unzip the model from [Google Drive](https://drive.google.com/file/d/1vfSIK_xSh-GY2GeW1_JGcrAba8mdZxpD/view?usp=share_link). Then, open the [```synthetic_data_generation.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/synthetic_data_generation.ipynb) notebook, update the path to the model, and run all cells.

## Speech corpus scraping
To automatically extract speech commands from a large-scale speech corpus, we used [Vosk Speech Recognition Toolkit](https://github.com/alphacep/vosk-api/tree/master). The example code is given in [```speech_corpus_scraping.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/speech_corpus_scraping.ipynb) notebook. 
 
## Data augmentation 
To increase the dataset size further, you can apply audio augmentation methods to the synthetic dataset and also to the speech corpus scraped dataset. The details can be found in the [```data_augmentation.ipynb```](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/blob/main/data_augmentation.ipynb) notebook.

## Model training, validation, and testing
The details of training, validation, and testing of the model can be found in the [Keyword-MLP](https://github.com/IS2AI/Kazakh-Speech-Commands-Dataset/tree/main/Keyword-MLP) directory. 

## Tutorials
Video tutorials for each step of the project on [our YouTube channel](https://www.youtube.com/playlist?list=PLYwixe_5vr_kpH8_iLBSs4hWDN-50GgGo)

## Citation
```
@INPROCEEDINGS{10601292,
  author={Kuzdeuov, Askat and Nurgaliyev, Shakhizat and Turmakhan, Diana and Laiyk, Nurkhan and Varol, Huseyin Atakan},
  booktitle={2023 3rd International Conference on Robotics, Automation and Artificial Intelligence (RAAI)}, 
  title={Speech Command Recognition: Text-to-Speech and Speech Corpus Scraping Are All You Need}, 
  year={2023},
  volume={},
  number={},
  pages={286-291},
  keywords={Accuracy;Speech coding;Virtual assistants;Speech recognition;Data collection;Benchmark testing;Data models;Speech commands recognition;text-to-speech;Kazakh Speech Corpus;voice commands;data-centric AI},
  doi={10.1109/RAAI59955.2023.10601292}}

```
