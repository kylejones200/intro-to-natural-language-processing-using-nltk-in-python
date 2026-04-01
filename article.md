# Intro to Natural Language Processing using NLTK in Python

This simple project shows how to use natural language processing to look
at the most common terms in a text.

::::### Intro to Natural Language Processing using NLTK in Python 

This simple project shows how to use natural language processing to look
at the most common terms in a text.


<figcaption>Photo by <a
href="https://unsplash.com/@purejulia?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com/@purejulia?utm_source=medium&amp;utm_medium=referral"
rel="photo-creator noopener" target="_blank">pure julia</a> on <a
href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
rel="photo-source noopener" target="_blank">Unsplash</a></figcaption>


We use the \`requests\` module to get the book "Alice's Adventures in
Wonderland" and then parse the document with \`BeautifulSoup\`.

```python
from bs4 import BeautifulSoup
from requests.packages import urllib3
```

Project Gutenberg has a number of books that are easily accessible. I
will access the HTML version because we can use the tags to help us
parse the text.

``` 
target = "https://www.gutenberg.org/cache/epub/11/pg11-images.html"
r = urllib3.PoolManager().request('GET', target)
soup = BeautifulSoup(r.data, "html.parser")
```

We fetch the book and save it as an object called 'r'. BeautifulSoup
parses the book and saves the text as a .txt file.

``` 
text = soup.get_text()
file = open("Alices Adventures in Wonderland.txt","w") 
file.write(text) 
file.close()
```

We will use the NLTK (natural langaugage took kit) to help us look at
the text. NLTK is a large library and you must download all the
components in order to use it. If this is the first time you have used
NLTK, you need to start with:

```python
import nltk
nltk.download()
```

This will open a separate GUI where you can select what to download. I
recommend that you download everything. In this project, we will use the
"stopwords" file from NLTK which is a list of words that are not
important for our analysis such as "is", "by", and "the".

```python
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
nltk.download('stopwords')
from nltk.corpus import stopwords
import re #regular expressions module
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
```

```python
filename = 'Alices Adventures in Wonderland.txt' #text of Alice's Adventures in Wonderland - from Project Gutenberg
file = open(filename, 'rt') #open and reads file to memory
text = file.read().lower() #converts all text to lower case
file.close()
# load text
```

Now wer have an object called "text" that has all the text of the book.
We want to "tokenize" the text and break it into individual words. Then
we remove all the words that are listed in the "stopwords" list from
NLTK using a [list
comprehension](https://www.w3schools.com/python/python_lists_comprehension.asp).

``` 
stopWords = set(stopwords.words('english')) #sets words to be removed like 
tokenizer = RegexpTokenizer(r'\w+') #removes punctuation
words=tokenizer.tokenize(text) #splits text by word 
wordsFiltered = []
 
wordsFiltered.append([w for w in words if w not in stopWords]) #loops over words and removes words that apear on the "stopWords" list
 
print(wordsFiltered[:100])
```

\`\`\`

\['project', 'gutenberg', 'ebook', 'alice', 'adventures', 'wonderland',
'lewis', 'carroll', 'project', 'gutenberg', 'ebook', 'alice',
'adventures', 'wonderland', 'ebook', 'use', 'anyone', 'anywhere',
'united', 'states', 'parts', 'world', 'cost', 'almost', 'restrictions',
'whatsoever', 'may', 'copy', 'give', 'away', 'use', 'terms', 'project',
'gutenberg', 'license', 'included', 'ebook', 'online', 'www',
'gutenberg', 'org', 'located', 'united', 'states', 'check', 'laws',
'country', 'located', 'using', 'ebook', 'title', 'alice', 'adventures',
'wonderland', 'author', 'lewis', 'carroll', 'release', 'date', 'june',
'27', '2008', 'ebook', '11', 'recently', 'updated', 'march', '30',
'2021', 'language', 'english', 'credits', 'arthur', 'dibianca', 'david',
'widger', 'start', 'project', 'gutenberg', 'ebook', 'alice',
'adventures', 'wonderland', 'alice', 'adventures', 'wonderland',
'lewis', 'carroll', 'millennium', 'fulcrum', 'edition', '3', '0',
'contents', 'chapter', 'rabbit', 'hole', 'chapter', 'ii', 'pool'\]

Now we can look at the term frequency of the filtered word list.

``` 
fd = nltk.FreqDist(wordsFiltered)
fd.plot(30,cumulative=False)
```


The most common word is "said" followed by "Alice". "Gutenberg" shows up
a lot because of the legal disclaimer at the end of the text.

### Sentiment Analysis using VADER with NLTK in Python
This project shows how to use
[VADER](https://www.nltk.org/api/nltk.sentiment.vader.html) to identify the sentiment of product
reviews. VADER does a better job of dealing with slang that other
sentiment analysis tools.

```python
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
```

We can use a simple example to see how VADER works.

``` 
text = 'This was a good movie.'
vader_scores = sid.polarity_scores(text)
print(vader_scores)
```

``` 
# Output:
```

The data comes from more than 34,000 customer reviews available on
[Kaggle](https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products/version/5?select=1429_1.csv). We use the rating, text, and title
columns. Like all project, we begin by importing pandas and the data.

```python
import pandas as pd
df = pd.read_csv("data/review_data.csv") 
```

``` 
df.head()
```


There are two ways to apply this code to each item in the data frame. \
1. Using \`.apply()\` and an anonymous function (\`lambda\`).
Traditional approach to applying a function to a data frame. \
2. List comprehension. Slightly faster and more "Pythonic".

In both cases, we need to convert all the text in the column to \`str\`.

``` 
%%time
# Option 1
df['scores_apply'] = df["reviews.text"].apply(lambda text: 
# Option 2
df['scores_list_comprehension'] = [sid.polarity_scores(str(i)) for i in df["reviews.text"]]
df.head()
```


We now have two new columns that contain a dictionary of the Vader
Polarity scores. We need to extract the dictionary and move this into
columns so we can use the data using \`scores =
df\['scores'\].apply(pd.Series)\` then use \`pd.concat()\` to join the
\`scores\` data frame with the original data frame.

``` 
df = pd.concat([df, df['scores_list_comprehension'].apply(pd.Series)], axis=1)
df["Category"] = df["reviews.rating"].astype("category")
df.head()
```


Now we have a lot more info we can use to explore the reviews than just
the start ratings.

Another way of using NLP is through AI Services available from different
cloud providers. This project uses the NLP services from AWS to look at
geoscience documents.

[**Natural Language Processing for Geoscience using Amazon Comprehend**\
*The endless rows of scientific publications and technical documents hid
the critical geochronological details I
needed...*medium.com](https://medium.com/@kylejones_47003/natural-language-processing-for-geoscience-using-amazon-comprehend-ec72c95d9c59 "https://medium.com/@kylejones_47003/natural-language-processing-for-geoscience-using-amazon-comprehend-ec72c95d9c59")[](https://medium.com/@kylejones_47003/natural-language-processing-for-geoscience-using-amazon-comprehend-ec72c95d9c59)
### Related Stories
- [[Learning Python by building a Simple Guessing
  Game](https://medium.com/@kylejones_47003/learning-python-by-building-a-simple-guessing-game-86d4f41fc2b8)]
- [[Predictive Maintenance using Principal Component Analysis in
  python](https://medium.com/@kylejones_47003/predictive-maintenance-using-principal-component-analysis-in-python-43a33538fcb2)]
- [[Survival Analysis for predictive maintenance with data from NASA
  Turbofan example in
  Python](https://medium.com/@kylejones_47003/survival-analysis-with-data-from-nasa-turbofan-example-in-python-0335e3338848)]
::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[December 20, 2023](https://medium.com/p/ddce6a0ff8ac).

[Canonical
link](https://medium.com/@kyle-t-jones/intro-to-natural-language-processing-using-nltk-in-python-ddce6a0ff8ac)

Exported from [Medium](https://medium.com) on November 10, 2025.
