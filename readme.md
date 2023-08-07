# MobileAnoNet

This project is an Anomaly Detection system based on the method proposed in *Decoupling Detectors for Scalable AnomalyDetection in AIoT Systems with Multiple Machines*, which has been submitted to Globecom2023 conference.

The code in the repository can be used to train a MobileAnoNet with DCASE datasets.

## Preparation
To download this project, use `git clone https://github.com/hqj-les30/MobileAnoNet.git`. Then switch to the project by `cd MobileAnoNet`.

To start training, you need to download DCASE22|DCASE23 data first. You can use `bash DL22/23.sh` for linux or download the data manually. **Notice**: to run the code, path to store the data have to be correct.

For the first time running this project, you also need to transform transform audio files to spectrum feature first. Run

```python src/feature_transform.py --set [DCASE22|DCASE23]```

## Training

After preparation, you can start training the model. Run

```python src/train_classification.py --device_ids id1,id2,... --mt_train machine1,machine2,... --mt_test machine1,machine2,... --data [DCASE22|DCASE23] --modelpath experiments/exp --bs 160```

If the number of *device_ids* is more than one, the model will be trained with DDP, where *bs* is the batch_size per GPU. You can change which or how many machines you want to train with. Or you can use "all" to represent all the machines. *modelpath* is the path to save checkpoints and result.