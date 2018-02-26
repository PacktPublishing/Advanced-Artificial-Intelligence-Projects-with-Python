import wikipedia
import spacy
from spacy.matcher import Matcher
import math
import re
from collections import Counter

nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

matched_phrases = []
def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start : end]
    matched_phrases.append(span.lemma_)
    
patterns = [[{'POS': 'NOUN', 'IS_ALPHA': True, 'IS_STOP': False, 'OP': '+'}]]
matcher = Matcher(nlp.vocab)
for pattern in patterns:
    matcher.add('keyword', collect_sents, pattern)

def extract_keywords_wikipedia(pagename, num_keywords):
    global matched_phrases
    page = wikipedia.page(pagename)
    pagenlp = nlp(page.content)
    matched_phrases = []
    matches = matcher(pagenlp)
    keywords = dict(Counter(matched_phrases).most_common(100))
    keywords_cvalues = {}
    for keyword in sorted(keywords.keys()):
        parent_terms = list(filter(lambda t: t != keyword and re.match('\\b%s\\b' % keyword, t), keywords.keys()))
        keywords_cvalues[keyword] = keywords[keyword]
        for pt in parent_terms:
            keywords_cvalues[keyword] -= float(keywords[pt])/float(len(parent_terms))
        keywords_cvalues[keyword] *= 1 + math.log(len(keyword.split()), 2)
    best_keywords = []
    for keyword in sorted(keywords_cvalues, key=keywords_cvalues.get, reverse=True)[:num_keywords]:
        best_keywords.append([keyword, keywords_cvalues[keyword]])
    return best_keywords


print(extract_keywords_wikipedia("New York City", 10))
print(extract_keywords_wikipedia("Python (programming language)", 10))
print(extract_keywords_wikipedia("Artificial intelligence", 10))
print(extract_keywords_wikipedia("Computer science", 10))


