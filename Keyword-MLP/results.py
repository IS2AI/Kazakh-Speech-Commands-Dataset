from sklearn.metrics import classification_report, confusion_matrix
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os


def evaluate(preds_path, labels):
    # groundtruth and predicted 
    # labels
    y_true = []
    y_pred = []

    # opening a JSON file
    f = open(preds_path)

    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    # iterating through the json list
    # and adding true and predicted labels
    for t, p in data.items():
        t = t.split('/')[-2]
        y_true.append(t)
        y_pred.append(p)

    # closing file
    f.close()

    # generate the classification report
    print(classification_report(y_true,y_pred, digits=4))

    # generate a confusion matrix in %
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cmn = np.round(cmn,1)

    # plot the confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(14, 14))
    ax= plt.subplot()
    sns.heatmap(cmn, annot=True, ax = ax, fmt=".1f", linewidth=.1, 
                cmap='YlGn', cbar=False, square=True, linecolor='white')
    
    # labels, title, and ticks
    ax.set_xlabel('Predicted commands', fontsize=12)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(labels, fontsize=10)
    ax.xaxis.tick_bottom()
    ax.set_ylabel('Actual commands', fontsize=12)
    ax.yaxis.set_ticklabels(labels, fontsize=10)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--preds", type=str, required=True, help="Path to preds.json file")
    args = parser.parse_args()
    assert os.path.exists(args.preds), f"Could not find file {args.preds}"

    labels = ["backward", "forward", "right", "left", "down", "up", "go", "stop", "on", "off", "yes", "no", 
              "learn", "follow", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
              "bed", "bird", "cat", "dog", "happy", "house", "read", "write", "tree", "visual", "wow"]

    evaluate(args.preds, labels)