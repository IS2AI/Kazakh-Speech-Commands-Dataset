# Keyword-MLP

In this project, we used [Keyword-MLP](https://github.com/AI-Research-BD/Keyword-MLP) model to develop a Kazakh Speech Commands Recognition. We sincerely thank the authors for open sourcing the code. 

## Setup

```
pip install -r requirements.txt
```

## Dataset
The Kazakh Speech Commands Benchmark (KSCB) dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1b2K8cU5rTTQutpibx-cPUF4smXodohEt/view?usp=share_link). The recordings are 1 second duration and saved in a WAV format with a sampling rate of 16 kHz. In total, 119 participants (62 males, 57 females) from different regions of Kazakhstan took part in data collection. The collected dataset underwent a manual evaluation by moderators to remove any subpar samples, including incomplete or incorrect readings, as well as quiet or silent recordings. The statistics of the collected dataset are provided below.

|ID| Command (kk)|Command (en)|# samples|
|--|--------|--------|---|
|1| backward | артқа | 113 |
|2| forward	| алға | 112  |
|3| right	| оңға | 106 | 
|4| left | солға | 104 | 
|5| down | төмен | 102 |
|6| up	 | жоғары | 104 | 
|7| go	 | жүр  | 101 |
|8| stop | тоқта | 107 |
|9| on	| қос	| 101 |
|10| off	| өшір	| 105 |
|11| yes	| иә | 110 |
|12| no	| жоқ	| 107 |
|13| learn | үйрен | 108 |	
|14| follow	| орында | 104 |
|15| zero	| нөл	| 105 |
|16| one	| бір	| 107 |
|17| two	| екі	| 99 |
|18| three	| үш | 107 |
|19| four	| төрт | 97 |
|20| five	| бес	| 104 |
|21| six	| алты | 101 |	
|22| seven	| жеті | 103 |
|23| eight	| сегіз	| 103 |
|24| nine	| тоғыз	| 100 |
|25| bed	| төсек	| 97 |
|26| bird	| құс	| 96 |
|27| cat	| мысық	| 97 |
|28| dog	| ит | 102 |
|29| happy	| бақытты	| 101 |
|30| house	| үй | 107 |
|31| read	| оқы	| 105 |
|32| write	| жаз	| 105 |
|33| tree	| ағаш | 104 |
|34| visual |	көрнекі	| 100 |
|35| wow	| мәссаған	| 99|



## Training

To train the model on the synthetically generated (Text-To-Speech) dataset + scraped Kazakh Speech Corpus 2 dataset, download the combined dataset from [Google Drive](https://drive.google.com/file/d/1tMiXB5vWqn8RgrmvCXCj-NVuDLEQ1E2e/view?usp=share_link) and unzip it inside the Keyword-MLP folder, then update the configuration file ```configs/kwmlp_ksc_tts.yaml``` and run the following script on your terminal:

```
python train.py --conf configs/kwmlp_ksc_tts.yaml
```

As an alternative option, you can use ```train_kscd.ipynb``` notebook.


## Testing

To test the pre-trained model (or a model you trained) on the KSCB dataset, download and unzip the dataset, and run the following scripts:
```
python inference.py --conf configs/kwmlp_ksc_tts.yaml \
                    --ckpt runs/kw-mlp-0.2.0-ksc-tts/best.pth \
                    --inp data/ \
                    --out outputs/ksc_tts/ \
                    --lmap label_map.json \
                    --device cpu \  # use cuda if you have a GPU
                    --batch_size 32 # should be possible to use much larger batches if necessary, like 128, 256, 512 etc.
```
```
python results.py --preds outputs/ksc_tts/preds.json
```
The last script outputs a classification report and a confusion matrix. As an alternative option, you can use ```test_kscd.ipynb```
```
              precision    recall  f1-score   support

    backward     0.9397    0.9646    0.9520       113
         bed     0.9684    0.9485    0.9583        97
        bird     0.7642    0.8438    0.8020        96
         cat     0.9314    0.9794    0.9548        97
         dog     0.6497    1.0000    0.7876       102
        down     0.8972    0.9412    0.9187       102
       eight     0.9802    0.9612    0.9706       103
        five     0.9894    0.8942    0.9394       104
      follow     0.9083    0.9519    0.9296       104
     forward     0.8750    0.8750    0.8750       112
        four     0.9487    0.7629    0.8457        97
          go     0.8349    0.9010    0.8667       101
       happy     0.9688    0.9208    0.9442       101
       house     1.0000    0.5607    0.7186       107
       learn     0.8750    0.9074    0.8909       108
        left     0.8704    0.9038    0.8868       104
        nine     0.9783    0.9000    0.9375       100
          no     1.0000    0.9065    0.9510       107
         off     1.0000    0.9143    0.9552       105
          on     0.7818    0.8515    0.8152       101
         one     0.9195    0.7477    0.8247       107
        read     0.6757    0.9524    0.7905       105
       right     0.8351    0.7642    0.7980       106
       seven     1.0000    0.8932    0.9436       103
         six     0.9314    0.9406    0.9360       101
        stop     0.9417    0.9065    0.9238       107
       three     1.0000    0.9065    0.9510       107
        tree     0.9804    0.9615    0.9709       104
         two     0.8936    0.8485    0.8705        99
          up     0.9677    0.8654    0.9137       104
      visual     0.9091    1.0000    0.9524       100
         wow     0.9510    0.9798    0.9652        99
       write     0.9043    0.9905    0.9455       105
         yes     0.8189    0.9455    0.8776       110
        zero     0.9175    0.8476    0.8812       105

    accuracy                         0.8979      3623
   macro avg     0.9088    0.8982    0.8984      3623
weighted avg     0.9089    0.8979    0.8983      3623
```
To test the model on randomly selected subsets of the KSCD, you can use ```test_kscd.ipynb``` notebook. 
