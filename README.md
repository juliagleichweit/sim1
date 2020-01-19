
# sim1
TU Wien Similarity Modeling 1 - Finding Kermit

#### Group members: 
Gleichweit Julia - 01325844
Häcker Benedikt - 11713128

### Prepare Files
Place your video files in the provided data directory.

Go to __src/util__ and set *obtainImg*, *genLabels* and *movFiles* in __prep_files_for_model.py__.
Default is True - False - True.    

*obtainImg=True*: The frames are stored in *processed_img/<video_file>* with 1 frame per second. You can then either 
* sort frames manually in the kermit and not_kermit folders and generate labels
* use the pre-sorted label files in *src/util*
sorting the 

*moveFiles=True*: according to the labels (i.e sorting of the frames) they are moved to __data/train__ and __data/test__. 
On default Muppets-02-01-01 and Muppets-03-04-03 are used for training and Muppets-02-04-04 for testing.  
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
