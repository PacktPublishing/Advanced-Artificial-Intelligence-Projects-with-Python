
import mailbox

mbox = mailbox.mbox('enron.mbox')

import re
def cleanup_email(msgbody):
    regex_replies = re.compile('(\-+Original Message|\-\-+|~~+).*', re.DOTALL) # find 'Original Message...' and variants
    msgbody = re.sub(regex_replies, '', msgbody)
    msgbody = re.sub(r'=\d\d', ' ', msgbody) # remove funny email formatting issues
    msgbody = re.sub(r'\s*>.*', '', msgbody) # remove quotes
    msgbody = re.sub(r'https?://.*?\s', '', msgbody) # remove links
    bigspace = re.compile(r'\n\n\n\n\n+.*', re.DOTALL) # find large gaps
    msgbody = re.sub(bigspace, '', msgbody)
    bigindent = re.compile(r'(\t| {4,}).*', re.DOTALL) # find big indentations (i.e., a quoted document)
    msgbody = re.sub(bigindent, '', msgbody)
    emailpaste = re.compile(r'(From|Subject|To): .*', re.DOTALL) # find pasted emails
    msgbody = re.sub(emailpaste, '', msgbody)
    msgbody = re.sub(r'=(\s*)\n', '\1', msgbody) # fix broken newlines
    msgbody = re.sub(r' ,([stm])', '\'\1', msgbody) # fix funny apostrophe 's and 't and 'm
    msgbody = re.sub(r'([\?\.])\?', '\1', msgbody) # fix funny extra question marks
    msgbody = re.sub(r'\x01', ' ', msgbody) # fix odd spaces
    return msgbody.strip()


# filter out announcement and mailing list emails, and emails to large numbers of people
def check_good_tofrom(msg):
    return (re.match(r'.*(admin|newsletter|list|announce|all[\._]|everyone[\.\_]).*', msg['From'], re.IGNORECASE) is None and
            msg['To'] is not None and
            re.match(r'.*(admin|newsletter|list|announce|all[\._]|everyone[\.\_]).*', msg['To'], re.IGNORECASE) is None and
            re.match(r'.*@enron\.com', msg['From'], re.IGNORECASE) and
            len(msg['To'].split()) <= 3)


import spacy
nlp = spacy.load('en')

from spacy.symbols import nsubj, xcomp, dobj, pobj, prep, attr, VERB, PRON, NOUN, PROPN, PUNCT

# simple method for anaphora resolution
def find_referent(doc, pronoun, msgfrom, msgto):
    # if pronoun is 'I' or 'myself' or 'me' or 'we', use msg from name
    if pronoun.text.lower() in ['i', 'myself', 'me', 'we']:
        return msgfrom
    # if pronoun is 'you' or 'your', use msg to name
    elif pronoun.text.lower() == 'you' or pronoun.text.lower() == 'your':
        return msgto
    # else, find root verb for pronoun, then find the subject
    else:
        w = pronoun
        while w.head != w: # walk up the tree to root
            w = w.head
        # now we have the root verb, find the nsubj if exists
        for c in w.children:
            if c.dep == nsubj:
                return c.text
        return None

def extract_relationships2(doc, msgfrom, msgto):
    #print(doc)
    relationships = []
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            subj = possible_subject
            verb = possible_subject.head
            
            if subj.pos == PRON or subj.pos == PROPN:
                if subj.pos == PRON:
                    ref = find_referent(doc, subj, msgfrom, msgto)
                    if ref is not None:
                        subj = ref
                    else:
                        subj = subj.text
                else:
                    subj = subj.text
                
                # ignore worthless subjects
                if subj.lower() in ['they', 'it']:
                    continue
                    
                for vr in verb.rights:
                    if vr.dep == xcomp:
                        for vc in vr.children:
                            if vc.dep == dobj and vc.pos == NOUN:
                                if vr.idx < vc.idx:
                                    relationships.append((subj, verb.lemma_, vr.lemma_ + " " + vc.lemma_))
                                else:
                                    relationships.append((subj, verb.lemma_, vc.lemma_ + " " + vr.lemma_))
                    elif vr.dep == prep:
                        for vc in vr.children:
                            if vc.dep == pobj and vc.pos == NOUN:
                                relationships.append((subj, verb.lemma_, vc.lemma_))
                    elif vr.dep == dobj and (vr.pos == NOUN or vr.pos == PROPN):
                        has_compound = False
                        for vc in vr.children:
                            if vc.dep_ == 'compound' and vc.pos == NOUN:
                                has_compound = True
                                if vr.idx < vc.idx:
                                    relationships.append((subj, verb.lemma_, vr.lemma_ + " " + vc.lemma_))
                                else:
                                    relationships.append((subj, verb.lemma_, vc.lemma_ + " " + vr.lemma_))
                        if not has_compound:
                            relationships.append((subj, verb.lemma_, vr.lemma_))
                    elif vr.dep == attr:
                        relationships.append((subj, verb.lemma_, vr.lemma_))
    return relationships

def extract_email_relationships(mbox, msg_key):
    message = mbox.get(msg_key)
    if message['From'] is not None and message['To'] is not None:
        try:
            msgnlp = nlp(cleanup_email(message.get_payload()))
            return extract_relationships2(msgnlp, message['From'], message['To'].split(', ')[0])
        except:
            return []
    else:
        return []

import rdflib
from rdflib import Graph, Literal, RDF
from rdflib.namespace import FOAF

def query_relationships(predicate, g, msg_key_idx, msg_key_idx_reverse):
    doc = nlp(predicate)
    p = Literal(doc[0].lemma_)
    qres = g.query('SELECT ?s ?o WHERE { ?s ?p ?o . }', initBindings = {'p' : p})

    for row in qres:
        r = (row[0], p, row[1])
        print("%s\t*%s*\t%s -- msg_keys: %s" % (row[0], p, row[1], msg_key_idx[r]))

def create_graph_from_email_relationships(mbox):
    g = Graph('Sleepycat', identifier='enron_relationships') # needs python lib bsddb3
    g.open('enron_relationships.rdf', create = True)
    msg_key_idx = {}
    msg_key_idx_reverse = {}
    
    i = 0
    msgs = mbox.keys()
    msg_count = len(msgs)
    for msg_key in msgs: # no limit now, do all messages
        i += 1
        if i % 10000 == 0:
            print("Message %d of %d" % (i, msg_count))
    
        # find relationships
        rels = extract_email_relationships(mbox, msg_key)
        
        # for each relationship
        for (s, p, o) in rels:
            
            r = (Literal(s), Literal(p), Literal(o))
            
            # add relationship to the graph
            g.add(r)
            
            # remember which message(s) had this relationship
            if r in msg_key_idx:
                msg_key_idx[r].append(msg_key)
            else:
                msg_key_idx[r] = [msg_key]
                
            # remember the relationships this message had
            if msg_key in msg_key_idx_reverse:
                msg_key_idx_reverse[msg_key].append(r)
            else:
                msg_key_idx_reverse[msg_key] = [r]
                
    return (g, msg_key_idx, msg_key_idx_reverse)

(g, msg_key_idx, msg_key_idx_reverse) = create_graph_from_email_relationships(mbox)

query_relationships("removed", g, msg_key_idx, msg_key_idx_reverse)



