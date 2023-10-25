# Configuration file

Here we provide a description of the main parameters on the configuration files. 

- `train_split`: Specifies the data split designated for model training.

- `val_split`: Specifies the data split designated for early stopping during training.

- `test_split`: Specifies the data split designated for final testing and evaluation.

- `chunk_size`: Specifies the duration of video clips, measured in seconds.

- `outputrate`: Specifies the prediction rate, the number of predictions generated at each second. 

- `audio`: Specifies the inclusion of audio information.

- `baidu`: Specifies the inclusion of baidu features.

- `path_labels`: Specifies the directory where action labels are stored.

- `path_baidu`: Specifies the directory where Baidu features are stored.

- `path_audio`: Specifies the directory where audio spectrograms are stored.

- `path_store`: Specifies the directory for storing clip samples extracted from original videos.

- `path_experiments`: Specifies the directory for storing model checkpoints, predictions, and related outputs.

- `store`: A binary parameter. Set to 1 to generate and store clip samples in the 1st execution, or set to 0 to only read them.

- `mixup`: Binary option for enabling or disabling mixup augmentation.

- `mixup_balanced`: When activated, signifies the preference for balanced mixup instead of the classic mixup approach.

- `uncertainty`: Binary option to choose between utilizing uncertainty-aware prediction heads or standard prediction heads.


