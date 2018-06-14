# some utility functions for plotting

def annotate_frequencies(ax, annotation_dict, x_off=1, line_length=0.2, higher_harm=1):
    """

    annotates the plot on axis ax

    Args:
        ax:
        annotation_dict: dictionary where keys are the text label and values are [x, y] coordinates
        higher_harm:

    Returns:

    """
    for k, (x, y) in annotation_dict.items():
        for hh in range(higher_harm):

            if hh == 0:
                text = k
            else:
                text = '2' + k
                x = 2 * x
                y += line_length

            ax.plot([x, x], [y - line_length, y], '-')
            ax.annotate(text, xy=(x, y), xytext=(x + x_off, y),arrowprops=None)
