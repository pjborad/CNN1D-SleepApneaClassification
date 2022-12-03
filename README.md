# CNN1D-SleepApneaClassification
- Obstructive Sleep Apnea (OSA) is a severe sleep disorder and it is identified by several ceases in breathing while sleeping, caused by insufficient oxygen at body and brain level. 
- The recognition of the OSA at primary stage can mitigate the chances of perilous health impairment. Thus, an accurate mechanized OSA detection system is imperative
- suffering of extreme sleep ailment is indicated by OSA based computer supported determination which is referred as Computer Aided Diagnosis (CAD) where changes in bio-signals can be detected. 
The CAD system can efficiently detect, cut cost, and classify at an early stage using single-channel bio-signals and it can be even implemented at home
- With the help of optimal orthogonal wavelet filter bank (OWFB),the 1-minute duration raw ECG are divided into six wavelet sub-bands (WSBs)
- The features such as Log-Energy (LGE), and Sample Entropy (SPE) are calculated for all six WSBs
- In this project, Convolutional Neural Network (CNN) with raw ECG signals, and Extreme Learning Machine (ELM) with LGE and SPE features based CAD systems are utilized for classification of normal and OSA
- The proposed systems are verified using various public open source data set such as MIT-BIH PSNY Database - 1999, CINC challenge - 2000, and St. Vincent's University Hospital Database (UCD) - 2007 
- The proposed OSA identification model demonstrate the 96.30%, 94.72% ,98.93% classification accuracy with OWFB features based ELM and 94.41%, 93.51%, 96.15% classification accuracy with CNN on all three datasets respectively
