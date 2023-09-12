import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


def plot_multi(scores1, mean_scores1, scores2, mean_scores2):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores1, label='Score 1')
    plt.plot(mean_scores1, label='Mean Score 1')
    plt.plot(scores2, label='Score 2')
    plt.plot(mean_scores2, label='Mean Score 2')
    plt.ylim(ymin=0)
    plt.text(len(scores1) - 1, scores1[-1], str(scores1[-1]))
    plt.text(len(mean_scores1) - 1, mean_scores1[-1], str(mean_scores1[-1]))
    plt.text(len(scores2) - 1, scores2[-1], str(scores2[-1]))
    plt.text(len(mean_scores2) - 1, mean_scores2[-1], str(mean_scores2[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
