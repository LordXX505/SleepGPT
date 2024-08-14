import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def calculate_metrics(conf_matrix):
    y_true = []
    y_pred = []
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            y_true.extend([i] * conf_matrix[i, j])
            y_pred.extend([j] * conf_matrix[i, j])

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    print(f1)
    return acc, mf1, kappa, f1

def main():
    confusion_matrix_raw = {
        1: np.array([[0, 0, 1675, 0, 0], [0, 0, 621, 0, 0], [0, 0, 4206, 0, 0], [0, 0, 850, 0, 0], [0, 0, 1848, 0, 0]]),
        2: np.array([[0, 0, 1675, 0, 0], [0, 0, 621, 0, 0], [0, 0, 4206, 0, 0], [0, 0, 850, 0, 0], [0, 0, 1848, 0, 0]]),
        5: np.array([[1559, 4, 43, 7, 62], [200, 18, 102, 5, 296], [23, 3, 3682, 237, 261], [2, 0, 103, 745, 0],
                     [67, 8, 240, 0, 1533]]),
        12: np.array([[1583, 50, 14, 2, 26], [176, 204, 93, 3, 145], [48, 86, 3696, 164, 212], [9, 0, 132, 709, 0],
                      [51, 80, 146, 0, 1571]])
    }

    confusion_matrix_aug = {
        1: np.array([[424, 0, 1118, 133, 0], [6, 0, 609, 6, 0], [187, 0, 3829, 190, 0], [692, 0, 69, 89, 0],
                     [0, 0, 1845, 3, 0]]),
        2: np.array([[1550, 0, 125, 0, 0], [223, 0, 398, 0, 0], [466, 0, 3740, 0, 0], [633, 0, 217, 0, 0],
                     [359, 0, 1489, 0, 0]]),
        5: np.array([[1525, 32, 35, 3, 80], [151, 74, 118, 3, 275], [31, 11, 3690, 179, 295], [2, 0, 185, 663, 0],
                     [14, 17, 154, 0, 1663]]),
        12: np.array([[1559, 57, 37, 3, 19], [151, 210, 123, 2, 135], [20, 46, 3839, 177, 124], [2, 0, 135, 713, 0],
                      [32, 83, 163, 0, 1570]])
    }

    metrics_raw = {n: calculate_metrics(cm) for n, cm in confusion_matrix_raw.items()}
    metrics_aug = {n: calculate_metrics(cm) for n, cm in confusion_matrix_aug.items()}

    subject_counts = sorted(metrics_raw.keys())
    metric_names = ['ACC', 'MF1', 'Kappa'] + [f'Stage{i + 1}_F1' for i in range(5)]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), sharey=True)
    axes = axes.flatten()

    width = 0.35  # width of the bars

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        if i < 3:
            raw_values = [metrics_raw[n][i] for n in subject_counts]
            augmented_values = [metrics_aug[n][i] for n in subject_counts]
            for n in subject_counts:
                print(f'n: {n} augmented_values-raw_values: {metrics_aug[n][i]-metrics_raw[n][i]}')
        else:
            stage_idx = i - 3
            raw_values = [metrics_raw[n][3][stage_idx] for n in subject_counts]
            augmented_values = [metrics_aug[n][3][stage_idx] for n in subject_counts]

        x = np.arange(len(subject_counts))

        ax.bar(x - width/2, raw_values, width, label='Raw Data' if i == 0 else "")
        ax.bar(x + width/2, augmented_values, width, label='Augmented Data' if i == 0 else "")

        ax.set_title(f'{metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(subject_counts)
        if i % 2 == 0:
            ax.set_ylabel('Score')
        if i >= len(metric_names) - 2:
            ax.set_xlabel('Number of Subjects')

    axes[0].legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    fig.suptitle('Comparison of Raw Data and Augmented Data across Different Subject Counts')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../result/portion.svg')

    plt.show()



if __name__ == "__main__":
    main()
