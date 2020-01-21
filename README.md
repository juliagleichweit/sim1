
# sim1
TU Wien Similarity Modeling 1 - Finding Kermit

#### Group members: 
Gleichweit Julia - 01325844
Häcker Benedikt - 11713128

### Prepare Files - Images
Place your video files in the provided data directory.

Go to __src/util__ and set *obtainImg*, *genLabels* and *movFiles* in __prep_files_for_model.py__.
Default is True - False - True.    

*obtainImg=True*: The frames are stored in *processed_img/<video_file>* with 1 frame per second. You can then either 
* sort frames manually in the kermit and not_kermit folders and generate labels
* use the pre-sorted label files in *src/util*


*moveFiles=True*: according to the labels (i.e sorting of the frames, Ground Truth (GT)) they are moved to __data/train__ and __data/test__. 
On default Muppets-02-01-01 and Muppets-03-04-03 are used for training and Muppets-02-04-04 for testing.  


### Prepare Files - Audio

I used ffmpeg to get the audio track and then to split the files in chunks:

>ffmpeg -i Muppets-03-04-03.avi -c copy -map 0:a Muppets-03-04-03.wav

>ffmpeg -i Muppets-02-01-01.avi -c copy -map 0:a Muppets-02-01-01.wav

>ffmpeg -i Muppets-02-04-04.avi -c copy -map 0:a Muppets-02-04-04.wav

To get the chunks (max. 1s long): 

> ffmpeg -i Muppets-03-04-03/Muppets-03-04-03.wav -c copy -map 0 -segment_time 1 -f segment Muppets-03-04-03/Muppets-03-04-03_%03d.mp4

> ffmpeg -i Muppets-02-01-01/Muppets-02-01-01.wav -c copy -map 0 -segment_time 1 -f segment Muppets-02-01-01/Muppets-02-01-01_%03d.mp4

> ffmpeg -i Muppets-02-04-04/Muppets-02-04-04.wav -c copy -map 0 -segment_time 1 -f segment Muppets-02-04-04/Muppets-02-04-04_%03d.mp4

In __data/audio__ you can find the corresponding labels for kermit and non_kermit. 
The __CNN_audio.py__ uses a Numpy-array to feed the network. You can produce these files via __src/preprocessing_audio.py__ or use the train (*trainchunks.npy*) and test file (*testchunks.npy*) are directly located in data.

In summary, this is our directory structure:
```
data/
    trainchunks.npy
    testchunks.npy
    
    audio/
        audioMuppets-02-01-01_labels_kermit.txt
        audioMuppets-02-01-01_labels_not_kermit.txt
        audioMuppets-02-04-04_labels_kermit.txt
        audioMuppets-02-04-04_labels_not_kermit.txt
        audioMuppets-03-04-04_labels_kermit.txt
        audioMuppets-03-04-04_labels_not_kermit.txt
    train/        
        kermit/
            <frame01>.jpg
            <frame03>.jpg
            ...
        not_kermit/
            <frame02>.jpg
            <frame07>.jpg
            ...                        
    test/        
        kermit/
            <frame01>.jpg
            <frame03>.jpg
            ...
        not_kermit/
            <frame02>.jpg
            <frame07>.jpg
            ...                         
    
```

After the above steps are done the test environment should be correctly set up.

#### Train the Models
Aftwerwards execute CNN_XXX.py and CNN_audio.py (a bit misleading name with C instead of D) to train the models or to test them on the prepared test data.
Therefore you have to set the shouldTrain and shouldTest boolean values in the corresponding script.
   
Both models employ a binary classifier. 

- For the audio 40 MFCCs (Mel-Frequency Cepstral Coefficient) are used. MFCCs make us of the  Mel scaling to try to model the way that the human hearing audiotory system perceives sounds.
That is also why we chose to use this features.

__AUDIO__: For the audio 40 MFCCs (Mel-Frequency Cepstral Coefficient) are used. MFCCs make us of the  Mel scaling to try to model the way that the human hearing audiotory system perceives sounds.
That is also why we chose to use this features.

The model is a simple deep neural network which should achieve around 83% accuracy on the test data.


#### Time Sheet: 
| Date| Time   | Julia | Benedikt| Description|
|-------|:---------|:-------|:----|:------|
|2019/10/08| 2.5h | X | X | Attended lecture|
|2019/10/11| 2.5h | X | X |Attended lecture|
|2019/10/17| 2.5h | X | X |  Attended lecture|
|2019/10/18| 2.5h | X | X |  Attended lecture|
|2019/10/18|2h|X | X| Pre-meeting and set-up-discussion
|2019/10/21 |  1h | | X | Set up environment|
|2019/10/20| 0.5h | X | |  Create github repository|
|2019/11/05| 1h | X | X| Research on possible neural network models for Kermit identification|
|2019/12/10| 4h |  | X| video to frames, frames to number transformation|
|2019/12/11| 4h | | X| labeling of frames|
|2019/12/12| 4h |  | X| CNN model for image classification|
|2019/12/13| 5h | X | | video to frames + labeling (1x 2FPS, 1x 1FPS)
|2019/12/14| 4h | X | | Inform about VGG16 and try to extract features for frames|
|2019/12/15| 4h |  | X| Audio to mfcc, first Dense Network, ROC curve|
|2019/12/15| 6h | X | | adapt Benedikt's model; build VGG16 like CNN and try transfer learning. try to optimize network, batch_sizes, etc. but I do not have enough memory to use VGG16 model or weights|
|2019/12/16| 8h | X | | network optimization to get higher accuracy on test data (1 epoch ~30min)|
|2020/01/16| 3h |  | X| Labeling of audio, CNN network for audio classification|
|2020/01/17| 6h |  | X| Training on GPU and crashed python|
|2020/01/17| 3h | X | | separate audio labeling (1s chunks), use ffmpeg to get audiotrack and 1s-chunks
|2020/01/19| 4h |  | X| Package fix, training|
|2020/01/20| 2h | X | | different audio classification|
|2020/01/20| 8h | X | X| image+audio classification in one, round up|

