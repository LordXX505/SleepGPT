import h5py
import numpy as np
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from timm.optim import adamw
from timm.scheduler import CosineLRScheduler
def set_random_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class SleepDisorderPipeline:
    def  __init__(self, h5_file_path, label_mapping, model_type='knn', num_channels=8, random_state=0, exclude_c=None,
                 train_subjects=None, test_subjects=None, reshape=False, select_stages=None, lr_rate=None):
        """
        Initialize the pipeline.
        """
        self.h5_file_path = h5_file_path
        self.label_mapping = label_mapping
        self.model_type = model_type
        self.num_channels = num_channels
        self.exclude_c = exclude_c if exclude_c else [5, 6, 7]
        self.models = {}
        self.confusion_matrices = []
        self.random_state = random_state
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects
        if model_type == 'nn':
            set_random_seed(random_state)
        self.reshape = reshape
        self.select_stages = select_stages
        if lr_rate is not None:
            assert model_type == 'nn'
            self.lr_rate = lr_rate
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SleepDisorderPipeline.NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, 2048)
            self.fc2 = nn.Linear(2048, 128)
            self.fc3 = nn.Linear(128, num_classes)
            self.relu1 = nn.GELU()
            self.relu2 = nn.GELU()

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.softmax(self.fc3(x))
            return x
        def predict_proba(self, x):
            return self(x)
    def split_subjects_fixed(self, test_size=0.2):
        """
        Split subjects using a fixed train-test split.
        """
        train_subjects, test_subjects = {}, {}
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            for pathology in h5_file.keys():
                subjects = list(h5_file[pathology].keys())
                train, test = train_test_split(subjects, test_size=test_size, random_state=self.random_state)
                train_subjects[pathology] = train
                test_subjects[pathology] = test
        return train_subjects, test_subjects

    def split_subjects_kfold(self, n_splits=4, random_state=512):
        """
        Split subjects using K-Fold cross-validation.
        """
        splits = []
        pathology_splits = {}

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            for pathology in h5_file.keys():
                subjects = np.array(list(h5_file[pathology].keys()))
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                pathology_splits[pathology] = []

                for train_idx, test_idx in kf.split(subjects):
                    train_subjects = subjects[train_idx].tolist()
                    test_subjects = subjects[test_idx].tolist()
                    pathology_splits[pathology].append((train_subjects, test_subjects))

        for fold in range(n_splits):
            train_subjects_fold = {}
            test_subjects_fold = {}

            for pathology, folds in pathology_splits.items():
                train_subjects, test_subjects = folds[fold]
                train_subjects_fold[pathology] = train_subjects
                test_subjects_fold[pathology] = test_subjects

            splits.append((train_subjects_fold, test_subjects_fold))

        return splits

    def organize_data_by_subjects(self, train_subjects, test_subjects, selected_stage, reshape=False, select_stages=None):
        """
        Organize data by subjects.
        """
        train_data = {channel: {} for channel in range(self.num_channels)}
        test_data = {channel: {} for channel in range(self.num_channels)}

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            for pathology in h5_file.keys():
                if pathology not in self.label_mapping.keys():
                    continue
                # print(f'{len(test_subjects[pathology])}, {test_subjects[pathology]}')
                for channel in range(self.num_channels):
                    if channel in self.exclude_c:
                        continue
                    if pathology not in train_data[channel]:
                        train_data[channel][pathology] = {}
                    if pathology not in test_data[channel]:
                        test_data[channel][pathology] = {}
                    for subject in h5_file[pathology].keys():
                        if selected_stage in h5_file[pathology][subject]:
                            dataset = h5_file[pathology][subject][selected_stage][:]
                            if reshape is True:
                                temp = rearrange(dataset, 'C (T D) -> T C D', T=5)
                                dataset = rearrange(temp[select_stages], 'T C D ->  C (T D) ')
                            if channel < dataset.shape[0] and np.sum(dataset[channel, :]) != 0:
                                channel_data = dataset[channel, :]
                            else:
                                channel_data = None

                            if pathology in train_subjects and subject in train_subjects[pathology]:
                                train_data[channel][pathology][subject] = channel_data
                            elif pathology in test_subjects and subject in test_subjects[pathology]:
                                test_data[channel][pathology][subject] = channel_data
                            else:
                                print(f'pathology: {pathology}, subject: {subject}')
                                raise RuntimeError
                # if pathology == 'nfle':
                #     for channel in range(self.num_channels):
                #         print(test_data[channel]['nfle'].keys())

        return train_data, test_data

    def train_channel_models(self, train_data, param_grid):
        """
        Train models for each channel.
        """
        for channel, pathologies in train_data.items():
            X_train, y_train = [], []
            for pathology, subjects in pathologies.items():
                for subject, channel_data in subjects.items():
                    if channel_data is not None:
                        X_train.append(channel_data)
                        y_train.append(self.label_mapping[pathology])

            if len(X_train) > 0:
                X_train, y_train = np.array(X_train), np.array(y_train)
                if self.model_type == 'knn':
                    model = KNeighborsClassifier()
                elif self.model_type == 'svm':
                    model = SVC(probability=True, class_weight='balanced')
                elif self.model_type == 'rf':
                    model = RandomForestClassifier()
                elif self.model_type == 'xgb':
                    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                elif self.model_type == 'logreg':
                    model = LogisticRegression(class_weight='balanced', max_iter=1000)
                elif self.model_type == 'gb':
                    model = GradientBoostingClassifier()
                elif self.model_type == 'nn':
                    input_dim = X_train.shape[1]
                    num_classes = len(np.unique(y_train))
                    model = self.NeuralNetwork(input_dim, num_classes)
                    optimizer = adamw.AdamW(model.parameters(), lr=self.lr_rate)
                    loss_fn = nn.CrossEntropyLoss()
                    X_train = torch.tensor(X_train, dtype=torch.float32)
                    y_train = torch.tensor(y_train, dtype=torch.long)
                    # scheduler = CosineLRScheduler(optimizer, t_initial=30)
                    for epoch in range(30):  # Example training loop
                        optimizer.zero_grad()
                        outputs = model(X_train)
                        loss = loss_fn(outputs, y_train)
                        loss.backward()
                        optimizer.step()
                        # scheduler.step(epoch=epoch)
                    self.models[channel] = {'model': model, 'classes': list(set(y_train.numpy()))}
                    continue
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=3,
                    verbose=1,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                print('finished searching')

                best_model = grid_search.best_estimator_
                self.models[channel] = {'model': best_model, 'classes': list(set(y_train))}


    def predict_with_channel_models(self, test_data):
        """
        Predict using trained models.
        """
        channel_predictions = defaultdict(list)
        ground_truth = defaultdict(list)

        for channel, pathologies in test_data.items():
            if channel not in self.models:
                continue

            model_info = self.models[channel]
            knn = model_info['model']
            classes = model_info['classes']
            for pathology, subjects in pathologies.items():
                if self.label_mapping[pathology] not in classes:
                    continue
                for subject, channel_data in subjects.items():
                    results = np.zeros(np.max(classes)+1)
                    if channel_data is not None:
                        if self.model_type == 'nn':
                            prediction = knn.predict_proba(torch.tensor(channel_data.reshape(1, -1), dtype=torch.float32))
                        else:
                            prediction = knn.predict_proba(channel_data.reshape(1, -1))
                        for index, c in enumerate(classes):
                            results[c] = prediction[0][index]
                    channel_predictions[(pathology, subject)].append(results)
                    ground_truth[(pathology, subject)].append(pathology)

        return channel_predictions, ground_truth

    def fuse_channel_predictions(self, channel_predictions, method="weighted"):
        """
        Fuse predictions across channels.
        """
        final_predictions = {}
        final_weighted_preds = {}
        for (pathology, subject), preds in channel_predictions.items():
            preds = np.stack(preds, axis=0)
            if method == "majority":
                votes = [np.argmax(proba) for proba in preds]
                final_predictions[(pathology, subject)] = Counter(votes).most_common(1)[0][0]
                weighted_preds = None
            elif method == "weighted":
                weights = np.ones(len(preds))
                weighted_preds = np.average(preds, axis=0, weights=weights)
                final_predictions[(pathology, subject)] = np.argmax(weighted_preds)
                final_weighted_preds[(pathology, subject)] = weighted_preds
        return final_predictions, final_weighted_preds

    def calculate_accuracy(self, final_predictions, ground_truth):
        """
        Calculate accuracy and confusion matrix.
        """
        y_pred = []
        y_true = []
        for key in final_predictions.keys():
            y_pred.append(final_predictions[key])
            y_true.append(self.label_mapping[ground_truth[key][0]])
        print(y_pred, y_true)
        acc = accuracy_score(y_pred, y_true)
        cm = confusion_matrix(y_true, y_pred)
        return acc, cm

    def run_pipeline(self, selected_stage, param_grid, split_method="fixed", n_splits=4):
        """
        Run the complete pipeline.
        """
        if self.train_subjects is not None:
            splits = [(self.train_subjects, self.test_subjects)]
        else:
            if split_method == "fixed":
                train_subjects, test_subjects = self.split_subjects_fixed()
                splits = [(train_subjects, test_subjects)]
            elif split_method == "kfold":
                splits = self.split_subjects_kfold(n_splits=n_splits)
            else:
                raise ValueError("Invalid split_method. Choose 'fixed' or 'kfold'.")
        st_confusion_matrices = []
        prob_matrices = {'pred': [], 'true': []}
        for fold_idx, (train_subjects, test_subjects) in enumerate(splits):
            print(f"\nRunning Fold {fold_idx + 1}/{len(splits)}")
            if isinstance(selected_stage, str):
                selected_stage = [selected_stage]
            stage_final_predictions = []
            for st in selected_stage:
                train_data, test_data = self.organize_data_by_subjects(train_subjects, test_subjects,
                                                                           st, self.reshape, self.select_stages)
                self.train_channel_models(train_data, param_grid)
                channel_predictions, ground_truth = self.predict_with_channel_models(test_data)
                final_predictions, final_weighted_preds = self.fuse_channel_predictions(channel_predictions, method="weighted")
                stage_final_predictions.append((final_weighted_preds, ground_truth))
                accuracy, cm = self.calculate_accuracy(final_predictions, ground_truth)
                self.confusion_matrices.append(cm)
                if len(selected_stage) == 1:
                    print(f"Fold {fold_idx + 1}, Accuracy: {accuracy:.2f}, CM:{cm}")
            if len(selected_stage)>1:
                res_pred = {}
                res_gt = {}
                for fpred, gt in stage_final_predictions:
                    for fpk, fpv in fpred.items():
                        if fpk not in res_pred.keys():
                            res_pred[fpk] = 0
                        res_pred[fpk] += fpv
                    for gtk, gtv in gt.items():
                        if gtk not in res_gt.keys():
                            res_gt[gtk] = gtv
                temp_pred_list = []
                temp_true_lsit = []
                for key in res_pred.keys():
                    temp_pred_list.append(res_pred[key])
                    temp_true_lsit.append(self.label_mapping[res_gt[key][0]])
                prob_matrices['pred'].extend(temp_pred_list)
                prob_matrices['true'].extend(temp_true_lsit)
                for k, v in res_pred.items():
                    res_pred[k] = np.argmax(v)
                # print(f"res_pred: {res_pred}")
                accuracy, cm = self.calculate_accuracy(res_pred, res_gt)
                st_confusion_matrices.append(cm)
                print(f"Fold {fold_idx + 1}, Accuracy: {accuracy:.2f}, CM:{cm}, prob_matrices: {prob_matrices}")
            # else:
#                 print('saving stage_final_predictions')
# #array([0.34000806, 0.65999192])
#                 np.save(f'/Users/hwx_admin/Sleep/result/UMAP/classification_apnea/apnea_{self.random_state}', stage_final_predictions, allow_pickle=True)
#                 print(f'stage_final_predictions: {stage_final_predictions}')
        if len(selected_stage) > 1:
            print("Overall Confusion Matrix:")
            print(np.sum(st_confusion_matrices, axis=0))
            print(f'Overall prob_matrices: {prob_matrices}')
            return np.sum(st_confusion_matrices, axis=0)
        else:
            print("Overall Confusion Matrix:")
            print(np.sum(self.confusion_matrices, axis=0))
            return np.sum(self.confusion_matrices, axis=0)
if __name__  == '__main__':
    confusion_matrix_res = {}
    param_grid_rf = {
        'n_estimators': [1000],
        'max_depth': [50],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'class_weight': ['balanced']
    }
    for random_state in range(0, 10):
        pipeline = SleepDisorderPipeline(
            h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
            label_mapping={'nfle': 0, 'rbd': 1, 'n': 2, },
            model_type='rf',
            random_state=random_state
        )
        confusion_matrixes = pipeline.run_pipeline(selected_stage=["0", "1", "2", "3", "4"], split_method="kfold",
                                                   n_splits=4, param_grid=param_grid_rf)
        confusion_matrix_res[random_state] = confusion_matrixes
    print(confusion_matrix_res)
    # for random_state in range(0, 10):
    #     pipeline = SleepDisorderPipeline(
    #         h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
    #         label_mapping={'nfle': 1, 'rbd': 2, 'n': 3, },
    #         model_type='rf',
    #         random_state=random_state
    #     )
    #     confusion_matrixes = pipeline.run_pipeline(selected_stage=["1", "2", "3", "4"], split_method="kfold",
    #                                                n_splits=4, param_grid=param_grid_rf)
    #     confusion_matrix_res[random_state] = confusion_matrixes
    # print(confusion_matrix_res)
    # for random_state in range(0, 10):
    #     pipeline = SleepDisorderPipeline(
    #         h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
    #         label_mapping={'nfle': 1, 'rbd': 2, 'n': 3, },
    #         model_type='rf',
    #         random_state=random_state
    #     )
    #     confusion_matrixes = pipeline.run_pipeline(selected_stage=["0", "2", "3", "4"], split_method="kfold",
    #                                                n_splits=4, param_grid=param_grid_rf)
    #     confusion_matrix_res[random_state] = confusion_matrixes
    # print(confusion_matrix_res)
    # for random_state in range(0, 10):
    #     pipeline = SleepDisorderPipeline(
    #         h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
    #         label_mapping={'nfle': 1,  'rbd': 2, 'n': 3, },
    #         model_type='rf',
    #         random_state=random_state
    #     )
    #     confusion_matrixes = pipeline.run_pipeline(selected_stage=["0", "1", "3", "4"], split_method="kfold", n_splits=4, param_grid=param_grid_rf)
    #     confusion_matrix_res[random_state] = confusion_matrixes
    # print(confusion_matrix_res)
    # for random_state in range(0, 10):
    #     pipeline = SleepDisorderPipeline(
    #         h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
    #         label_mapping={'nfle': 1, 'rbd': 2, 'n': 3, },
    #         model_type='rf',
    #         random_state=random_state
    #     )
    #     confusion_matrixes = pipeline.run_pipeline(selected_stage=["0", "1", "2", "4"], split_method="kfold",
    #                                                n_splits=4, param_grid=param_grid_rf)
    #     confusion_matrix_res[random_state] = confusion_matrixes
    # print(confusion_matrix_res)
    # for random_state in range(0, 10):
    #     pipeline = SleepDisorderPipeline(
    #         h5_file_path='../../../result/UMAP/CAP_umap/data.h5',
    #         label_mapping={'nfle': 1, 'rbd': 2, 'n': 3, },
    #         model_type='rf',
    #         random_state=random_state
    #     )
    #     confusion_matrixes = pipeline.run_pipeline(selected_stage=["0", "1", "2", "3"], split_method="kfold",
    #                                                n_splits=4, param_grid=param_grid_rf)
    #     confusion_matrix_res[random_state] = confusion_matrixes
    # print(confusion_matrix_res)