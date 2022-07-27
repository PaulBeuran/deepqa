import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud

class WordFrequencyColorizer():
  """
  Colorizer that use a word frequency dictionary and a scalar mappable to define a colorization
  for each word correlated to their respective frequency in respect of the scalar mappable's
  color map.

  Attributes
  ----------
  word_frequencies : dict
    Word frequencies dictionary
  scalar_mappable : matplotlib.cm.ScalarMappable
    Scalar mappable object that adapt some data scale to a color map and give RGBA values
    of scalar values according to that scale.

  Methods
  -------
  __call__(word, **kwargs):
    Return the hexadecimal colorization of a word by its frequency
  """

  def __init__(self, words_frequencies, scalar_mappable):
    """
    Constructs all the necessary attributes for the object.

    Parameters
    ----------
      word_frequencies : dict
        Word frequencies dictionary
      scalar_mappable : matplotlib.cm.ScalarMappable
        Scalar mappable object that adapt some data scale to a color map and give RGBA values
        of scalar values according to that scale.
    """
    self.words_frequencies = words_frequencies
    self.scalar_mappable = scalar_mappable
    self.scalar_mappable.set_array(np.array(list(words_frequencies.values())))
    self.scalar_mappable.autoscale()

  def __call__(self, word, **kwargs):
    """
    Return the hexadecimal colorization of a word by its frequency

    Parameters
    ----------
      word : str
        Word to colorize by its frequency
    
    Returns
    -------
    str
      An hexadecimal color code for the given word
    """
    word_rgba = self.scalar_mappable.to_rgba(self.words_frequencies[word])
    return mpl.colors.to_hex(word_rgba)


def top_tf_idf_wordcloud_ax(terms_tf_idf, ax, max_words, **wordcloud_kwargs):
    """
    Make a wordcloud subplot from a word-frequency dictionary

    Parameters
    ----------
    terms_tf_idf : dict
      Word-frequency dictionary
    ax : matplotlib.axes.Axes
      Axes to plot on the wordcloud
    max_words : int
      Maximum number of words in the wordcloud
    **wordcloud_kwargs : dict
      Additional wordcloud.Wordcloud constructor parameters (other than max_words) 

    Returns
    -------
    matplotlib.ax.Axes
      Axes given in parameters
    matplotlib.ax.AxesImage
      Resulting wordcloud AxesImage
    """
    wordcloud_freq_img = wordcloud.WordCloud(max_words=max_words, **wordcloud_kwargs)\
                                        .generate_from_frequencies(terms_tf_idf)
    img = ax.imshow(wordcloud_freq_img)
    ax.axis("off")   
    return ax, img


def top_tf_idf_wordcloud_plot(terms_tf_idf, 
                              max_words=200, 
                              figsize=(12, 6.75),
                              title=None,
                              title_fontsize=18,
                              scalar_mappable=mpl.cm.ScalarMappable(cmap=mpl.cm.terrain),
                              **wordcloud_kwargs):
  """
  Make a wordcloud figure from a word-frequency dictionary, using a word frequency colorizer
  to color the words according to their frequency

  Parameters
  ----------
  terms_tf_idf : dict
    Word-frequency dictionary
  max_words : int, default=200
    Maximum number of words in the wordcloud.
  figsize : 2-tuple of int, default=(12, 6.75)
    Figure size
  title : str, default=None
    Figure title. If None, output the default title
  title_fontsize : int, default=18
    Figure title fontsize
  scalar_mappable : matplotlibe.cm.ScalarMappable, default=ScalarMappable(cmap=mpl.cm.terrain),
    Scalar mappable used in the word frequency colorizer
  **wordcloud_kwargs : dict
    Additional wordcloud.Wordcloud constructor parameters (other than max_words) 

  Returns
    -------
  matplotlib.ax.Figure
    Wordcloud figure
  """
  if title is None:
    title = f"Top {max_words} TF-IDF wordcloud"
  fig, ax = plt.subplots(figsize=figsize)
  _, img = top_tf_idf_wordcloud_ax(terms_tf_idf, ax=ax, max_words=200,
                                   width=1600, height=900,
                                   color_func=WordFrequencyColorizer(terms_tf_idf, 
                                                                     scalar_mappable),
                                   **wordcloud_kwargs)
  cbar = fig.colorbar(scalar_mappable)
  cbar.set_label("TF-IDF frequency")
  fig.suptitle(title, fontsize=title_fontsize)
  return fig

def text_token_length_distribution_plot(corpus,
                                        tokenizer = re.compile(r"(?u)\b\w\w+\b").findall,
                                        figsize=(12, 6.75),
                                        title=None,
                                        title_fontsize=18):
  """
  Make a distribution figure of text length in terms defined by a tokenizer

  Parameters
  ----------
  terms_tf_idf : dict
    Word-frequency dictionary
  figsize : 2-tuple of int, default=(12, 6.75)
    Figure size
  title : str, default=None
    Figure title. If None, output the default title
  title_fontsize : int, default=18
    Figure title fontsize

  Returns
    -------
  matplotlib.ax.Figure
    Text length distribution figure
  """
  if title is None:
    title = "Text token length density & cumulative distribution"
  
  # TODO: factorize
  corpus_token_lengths = [len(tokenizer(text)) for text in corpus]
  corpus_token_lengths_describe = (pd.Series(corpus_token_lengths)
                                      .describe(percentiles=[.25, .5, .75, .9, .95, .999]))

  fig, ax = plt.subplots(figsize=figsize)
  sns.histplot(corpus_token_lengths, kde=True, ax=ax, color="b", element="step", label="Density (Count)")

  mean = corpus_token_lengths_describe["mean"]
  ax.axvline(mean, label=f"Mean: {mean}", color="black", linestyle="--")

  ax2 = ax.twinx()
  sns.ecdfplot(corpus_token_lengths, ax=ax2, color="r", label="Cumulative distribution")

  fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
  ax.grid(which="both", axis="both")
  fig.suptitle(title, fontsize=title_fontsize)
  return fig