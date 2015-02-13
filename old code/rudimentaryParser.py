'''
Get a subset of pages here (includes history)
All pages including history would amount to several terabytes.
http://en.wikipedia.org/w/index.php?title=Special:Export&action=submit
We should definitely start off using this!
First tries indicate a maximum of 1000 revisions per article.
(that should still give us enough data, BUT it only gives the first 1000 revisions)


Parsing the data:

Try 1: wikidump
https://github.com/saffsd/wikidump
https://pypi.python.org/pypi/wikidump/0.1.2
not useful for our case:
    - documentation is non existent
    - it can't process revisions
It offers some very useful snippets though that we might want to use:
    - analysis.py offers plots made in matplotlib
    - it uses this counter dict http://code.activestate.com/recipes/576611/
        don't know if this is useful
    - dataset.py offers code to remove non-content (anchors, templates etc)
    - model.py has the parsing part in it

Try 2: mwparserfromhell
http://mwparserfromhell.readthedocs.org/en/latest/integration.html
seems to have the same problem that wikidump has. No revisions, tags in the text
I see no advantage using this (also the documentation is lacking).
Code wise it offers very little of what we can use.

Try 3: wikipedia API
http://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsectionformat=plain&titles=Iron_Man&format=xml
You can get a plain text (no tags) version of a text.
It does however have html tags in it.
We can't get more than the current revision of it - don't use.

Try 4: Parse it with BeautifulSoup
We can use BeautifulSoup to parse the xml and use the code snippets from wikidump
documentation here: http://www.crummy.com/software/BeautifulSoup/bs4/doc/
Following example demonstrates the BS advantage (http://stackoverflow.com/questions/16533153/parse-xml-dump-of-wiki)

from bs4 import BeautifulSoup as BS
# given your html as the variable 'html'
soup = BS(html, "xml")
pages = soup.find_all('page')
for page in pages:
    if page.ns.text == '0':
        print page.title.text

It does however tend to slow down if the file is large which it is in our case.
-> You can use lxml within the code in order to make it faster.




'''

from bs4 import BeautifulSoup as BS
#from BeautifulSoup import BeautifulSoup as BS
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# open the xml file
soup = BS(open(os.path.join(os.getcwd(),"dump2.xml")), "lxml")

#a revision tag is a single revision
pages = soup.find_all('revision')

#stripped strings returns all string values within the children of a single tag
'''
test = pages[0].stripped_strings
for string in test:
    print repr(string)
print "\n"
'''
#returns an array with all children of a node
'''
test2 = pages[0].contents
print test2
print "\n"
'''
#find all text tags
'''
pages2 = soup.find_all('text')
texts = []
for a in pages2:
    texts.append(a.text)
print texts[0]

#plot some stuff
lengths = [len(a) for a in texts]
plt.plot(lengths)
plt.title("Length of the article over time")
plt.show()
'''
#plot the number of revisions per year
'''
dates = Counter()
stamps = soup.find_all('timestamp')
for stamp in stamps:
    dates[stamp.text[0:4]] = dates[stamp.text[0:4]] + 1

items = [(k, v) for k, v in dates.items()]
items.sort()

years, number = zip(*items)
plt.plot(years, number)
plt.show()
'''
