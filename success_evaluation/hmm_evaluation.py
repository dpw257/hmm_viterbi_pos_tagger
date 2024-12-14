'''
This notebook evaluates the corpuses tagged using unigrams and bigrams against the corpus tagged manually. 
The POS-tagger is assessed on accuracy, precision and f-score.
Confusion matrices are generated for POS tags denoting verb forms and for tags denoting markers of syntactic complexity, for both the unigram and bigram method. 
'''


# Import libraries
import pandas as pd

# Define method to open 
def parse_csv_to_tuples(filepath):
    """
    Reads a CSV file, extracts tagged data, and converts it into a list of lists of tuples.
    """
    df = pd.read_csv(filepath)
    single_str_list = df.values.tolist()
    table = [sent[1:] for sent in single_str_list]
    
    parsed_tuples = []
    for sent in table:
        new_sent = []
        for tag_pair in sent:
            if isinstance(tag_pair, str) and '_' in tag_pair:
                split_index = tag_pair.index('_')
                new_sent.append((tag_pair[:split_index], tag_pair[split_index + 1:]))
        parsed_tuples.append(new_sent)
    return parsed_tuples

def process_decades(decades, file_types):
    """
    Processes files tagged using unigrams, using bigrams, and the manually tagged answers for each decade, returning results as dictionaries.
    """
    results = {ftype: [] for ftype in file_types}
    
    for decade in decades:
        for ftype in file_types:
            filepath = f"testset_{decade}_{ftype}.csv"
            results[ftype].append(parse_csv_to_tuples(filepath))
    
    return results

# Define decades and file types
decades = ["1960", "1970", "1980", "1990", "2000", "2010"]
file_types = ["unigrams_tagged", "bigrams_tagged", "answers"]

# Process the data
data = process_decades(decades, file_types)

# Assign processed data to variables for convenience
unigram_files = data["unigrams_tagged"]
bigram_files = data["bigrams_tagged"]
answer_files = data["answers"]

# Example usage: checking lengths of 2000s and 2010s data
print(len(unigram_files[4]))  # 2000s unigrams
print(len(bigram_files[4]))   # 2000s bigrams
print(len(answer_files[4]))   # 2000s answers

print(len(unigram_files[5]))  # 2010s unigrams
print(len(bigram_files[5]))   # 2010s bigrams
print(len(answer_files[5]))   # 2010s answers




# TABLE 1 - Accuracy 

syntax_tags = ['comp', 'cop', 'rel', 'sconj']
verb_tags = ['be.con', 'be.con.aut', 'be.con.syn', 'be.fut', 'be.fut.aut', 'be.fut.spec', 'be.fut.syn', 'be.hab', 'be.hab.aut', 'be.hab.syn', 'be.ord', 'be.ord.aut', 'be.ord.syn', 'be.past', 'be.past.aut', 'be.past.syn', 'be.pasthab', 'be.pasthab.syn', 'be.pres', 'be.pres.aut', 'be.pres.syn', 'be.preshab', 'be.preshab.syn', 'be.sub', 'vb.con', 'vb.con.aut', 'vb.con.syn', 'vb.fut', 'vb.fut.aut', 'vb.fut.spec', 'vb.fut.syn', 'vb.inf', 'vb.ord', 'vb.ord.aut', 'vb.ord.syn', 'vb.org', 'vb.past', 'vb.past.aut', 'vb.past.hab', 'vb.past.spec', 'vb.past.syn', 'vb.pasthab', 'vb.pasthab.aut', 'vb.pasthab.syn', 'vb.pres', 'vb.pres.aut', 'vb.pres.spec', 'vb.pres.syn', 'vb.sub', 'vb.sub.aut', 'vb.sub.syn', 'vn']

# Unigrams
overall_accuracy = 0
syntax_total = 0
syntax_correct = 0
verb_total = 0
verb_correct = 0
print('UNIGRAMS')
print('Accuracy:')
for f in range(6):  
    unigram_total = 0
    unigram_correct = 0
    for s in range(len(unigram_files[f])):
        for i in range(len(unigram_files[f][s])):
            if unigram_files[f][s][i][1] == answer_files[f][s][i][1]:
                unigram_total+= 1
                unigram_correct+= 1
                if answer_files[f][s][i][1] in syntax_tags:
                    syntax_total+= 1
                    syntax_correct+= 1
                if answer_files[f][s][i][1] in verb_tags:
                    verb_total+= 1
                    verb_correct+= 1
            else:
                unigram_total+= 1
                if answer_files[f][s][i][1] in syntax_tags:
                    syntax_total+= 1
                if answer_files[f][s][i][1] in verb_tags:
                    verb_total+= 1
    print(unigram_correct/unigram_total*100)
    overall_accuracy+=unigram_correct/unigram_total*100

print('Overall:', overall_accuracy/6)
print('Syntax correct', syntax_correct/syntax_total*100)
print('Verb correct', verb_correct/verb_total*100, '\n')


# Bigrams
bigram_total = 0
bigram_correct = 0
overall_accuracy = 0
syntax_total = 0
syntax_correct = 0
verb_total = 0
verb_correct = 0
print('BIGRAMS')
print('Accuracy:')
for f in range(6):  
    bigram_total = 0
    bigram_correct = 0
    for s in range(len(bigram_files[f])):
        for i in range(len(bigram_files[f][s])):
            if bigram_files[f][s][i][1] == answer_files[f][s][i][1]:
                bigram_total+= 1
                bigram_correct+= 1
                if answer_files[f][s][i][1] in syntax_tags:
                    syntax_total+= 1
                    syntax_correct+= 1
                if answer_files[f][s][i][1] in verb_tags:
                    verb_total+= 1
                    verb_correct+= 1
            else:
                bigram_total+= 1
                if answer_files[f][s][i][1] in syntax_tags:
                    syntax_total+= 1
                if answer_files[f][s][i][1] in verb_tags:
                    verb_total+= 1
    print(bigram_correct/bigram_total*100)
    overall_accuracy+=bigram_correct/bigram_total*100

print('Overall:', overall_accuracy/6)
print('Syntax correct', syntax_correct/syntax_total*100)
print('Verb correct', verb_correct/verb_total*100)





# Confusion matrix for syntax marker tags - unigrams

syntax_index = ['comp', 'conj', 'cop', 'neg', 'rel', 'sconj', 'par', 'pos', 'prep', 'other']
syntax_columns = {'comp': 0, 'conj': 0, 'cop': 0, 'neg': 0, 'rel': 0, 'sconj': 0, 'par': 0, 'pos': 0, 'prep': 0, 'other': 0}



syntax_matrix = []
for n in range(len(syntax_index)):
    syntax_matrix.append(syntax_columns.copy())
    
    
for f in range(len(unigram_files)):
    for s in range(len(unigram_files[f])):
        for i in range(len(unigram_files[f][s])):
            if answer_files[f][s][i][1] in syntax_index and unigram_files[f][s][i][1] in syntax_index:
                loc = syntax_index.index(answer_files[f][s][i][1])
                key = unigram_files[f][s][i][1]
                syntax_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] in syntax_index and unigram_files[f][s][i][1] not in syntax_index:
                loc = syntax_index.index(answer_files[f][s][i][1])
                key = 'other'
                syntax_matrix[loc]['other']+=1
            elif answer_files[f][s][i][1] not in syntax_index and unigram_files[f][s][i][1] in syntax_index:
                loc = syntax_index.index('other')
                key = unigram_files[f][s][i][1]
                syntax_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] not in syntax_index and unigram_files[f][s][i][1] not in syntax_index:
                loc = syntax_index.index('other')
                key = 'other'
                syntax_matrix[loc]['other']+=10

syntax_df = pd.DataFrame(syntax_matrix, syntax_index)    
print('SYNTAX UNIGRAMS CONFUSION MATRIX:')
print(syntax_df, '\n')

# Calculating precision, recall, fscore
print('UNIGRAMS:')
print('SYNTAX: Precision - Recall - F-score')
for i in range(len(syntax_index)):
    tp = syntax_df.at[syntax_index[i],syntax_index[i]]
    fp = syntax_df[syntax_index[i]].sum()-syntax_df.at[syntax_index[i],syntax_index[i]]
    fn = syntax_df.iloc[i].sum()-syntax_df.at[syntax_index[i],syntax_index[i]]
    tn = syntax_df.at[syntax_index[i],syntax_index[i]]-syntax_df[syntax_index[i]].sum()-syntax_df.iloc[i].sum()
    for n in range(len(syntax_index)):
        tn += syntax_df.iloc[n].sum()
    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    fscore = 2*precision*recall/(precision+recall)
    print(syntax_index[i], round(precision, 3), round(recall, 3), round(fscore, 3))
print('\n')


syntax_matrix = []
for n in range(len(syntax_index)):
    syntax_matrix.append(syntax_columns.copy())
    
    
for f in range(len(bigram_files)):
    for s in range(len(bigram_files[f])):
        for i in range(len(bigram_files[f][s])):
            if answer_files[f][s][i][1] in syntax_index and bigram_files[f][s][i][1] in syntax_index:
                loc = syntax_index.index(answer_files[f][s][i][1])
                key = bigram_files[f][s][i][1]
                syntax_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] in syntax_index and bigram_files[f][s][i][1] not in syntax_index:
                loc = syntax_index.index(answer_files[f][s][i][1])
                key = 'other'
                syntax_matrix[loc]['other']+=1
            elif answer_files[f][s][i][1] not in syntax_index and bigram_files[f][s][i][1] in syntax_index:
                loc = syntax_index.index('other')
                key = bigram_files[f][s][i][1]
                syntax_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] not in syntax_index and bigram_files[f][s][i][1] not in syntax_index:
                loc = syntax_index.index('other')
                key = 'other'
                syntax_matrix[loc]['other']+=10
print('\n')



syntax_df = pd.DataFrame(syntax_matrix, syntax_index)    
print('SYNTAX BIGRAMS CONFUSION MATRIX:')
print(syntax_df, '\n')

# Calculating precision, recall, fscore
print('BIGRAMS:')
print('SYNTAX: Precision - Recall - F-score')
for i in range(len(syntax_index)):
    tp = syntax_df.at[syntax_index[i],syntax_index[i]]
    fp = syntax_df[syntax_index[i]].sum()-syntax_df.at[syntax_index[i],syntax_index[i]]
    fn = syntax_df.iloc[i].sum()-syntax_df.at[syntax_index[i],syntax_index[i]]
    tn = syntax_df.at[syntax_index[i],syntax_index[i]]-syntax_df[syntax_index[i]].sum()-syntax_df.iloc[i].sum()
    for n in range(len(syntax_index)):
        tn += syntax_df.iloc[n].sum()
    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    fscore = 2*precision*recall/(precision+recall)
    print(syntax_index[i], round(precision, 3), round(recall, 3), round(fscore, 3))
print('\n')




# Generate confusion matrices for tagging of verb forms

verbs_index = ['vb.pres', 'vn', 'vb.ord.syn', 'vb.past', 'vb.con.aut', 'vb.ord', 'vb.pasthab', 'vb.pasthab.aut', 'vb.past.aut', 'vb.con.syn', 'vb.fut', 'vb.sub', 'vb.pres.aut', 'vb.pres.spec', 'vb.pres.syn', 'vb.pasthab.syn', 'vb.past.syn', 'vb.ord.aut', 'vb.inf', 'vb.con', 'vb.sub.syn', 'vb.sub.aut', 'vb.fut.syn', 'vb.fut.aut', 'other']
verbs_columns = {'vb.pres': 0, 'vn': 0, 'vb.ord.syn': 0, 'vb.past': 0, 'vb.con.aut': 0, 'vb.ord': 0, 'vb.pasthab': 0, 'vb.pasthab.aut': 0, 'vb.past.aut': 0, 'vb.con.syn': 0, 'vb.fut': 0, 'vb.sub': 0, 'vb.pres.aut': 0, 'vb.pres.spec': 0, 'vb.pres.syn': 0, 'vb.pasthab.syn': 0, 'vb.past.syn': 0, 'vb.ord.aut': 0, 'vb.inf': 0, 'vb.con': 0, 'vb.sub.syn': 0, 'vb.sub.aut': 0, 'vb.fut.syn': 0, 'vb.fut.aut': 0, 'other': 0}

verbs_matrix = []
for n in range(len(verbs_index)):
    verbs_matrix.append(verbs_columns.copy())
    
    
for f in range(len(unigram_files)):
    for s in range(len(unigram_files[f])):
        for i in range(len(unigram_files[f][s])):
            if answer_files[f][s][i][1] in verbs_index and unigram_files[f][s][i][1] in verbs_index:
                loc = verbs_index.index(answer_files[f][s][i][1])
                key = unigram_files[f][s][i][1]
                verbs_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] in verbs_index and unigram_files[f][s][i][1] not in verbs_index:
                loc = verbs_index.index(answer_files[f][s][i][1])
                key = 'other'
                verbs_matrix[loc]['other']+=1
            elif answer_files[f][s][i][1] not in verbs_index and unigram_files[f][s][i][1] in verbs_index:
                loc = verbs_index.index('other')
                key = unigram_files[f][s][i][1]
                verbs_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] not in verbs_index and unigram_files[f][s][i][1] not in verbs_index:
                loc = verbs_index.index('other')
                key = 'other'
                verbs_matrix[loc]['other']+=10

verbs_df = pd.DataFrame(verbs_matrix, verbs_index)    



# Calculating precision, recall, fscore
print('UNIGRAMS:')
print('VERBS: Precision - Recall - F-score')
for i in range(len(verbs_index)):
    tp = verbs_df.at[verbs_index[i],verbs_index[i]]
    fp = verbs_df[verbs_index[i]].sum()-verbs_df.at[verbs_index[i],verbs_index[i]]
    fn = verbs_df.iloc[i].sum()-verbs_df.at[verbs_index[i],verbs_index[i]]
    tn = verbs_df.at[verbs_index[i],verbs_index[i]]-verbs_df[verbs_index[i]].sum()-verbs_df.iloc[i].sum()
    for n in range(len(verbs_index)):
        tn += verbs_df.iloc[n].sum()
    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    fscore = 2*precision*recall/(precision+recall)
    print(verbs_index[i], round(precision, 3), round(recall, 3), round(fscore, 3))
print('\n')


verbs_matrix = []
for n in range(len(verbs_index)):
    verbs_matrix.append(verbs_columns.copy())
    
    
for f in range(len(bigram_files)):
    for s in range(len(bigram_files[f])):
        for i in range(len(bigram_files[f][s])):
            if answer_files[f][s][i][1] in verbs_index and bigram_files[f][s][i][1] in verbs_index:
                loc = verbs_index.index(answer_files[f][s][i][1])
                key = bigram_files[f][s][i][1]
                verbs_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] in verbs_index and bigram_files[f][s][i][1] not in verbs_index:
                loc = verbs_index.index(answer_files[f][s][i][1])
                key = 'other'
                verbs_matrix[loc]['other']+=1
            elif answer_files[f][s][i][1] not in verbs_index and bigram_files[f][s][i][1] in verbs_index:
                loc = verbs_index.index('other')
                key = bigram_files[f][s][i][1]
                verbs_matrix[loc][key]+=1
            elif answer_files[f][s][i][1] not in verbs_index and bigram_files[f][s][i][1] not in verbs_index:
                loc = verbs_index.index('other')
                key = 'other'
                verbs_matrix[loc]['other']+=10

verbs_df = pd.DataFrame(verbs_matrix, verbs_index)    


# Calculating precision, recall, fscore
print('BIGRAMS:')
print('VERBS: Precision - Recall - F-score')
for i in range(len(verbs_index)):
    tp = verbs_df.at[verbs_index[i],verbs_index[i]]
    fp = verbs_df[verbs_index[i]].sum()-verbs_df.at[verbs_index[i],verbs_index[i]]
    fn = verbs_df.iloc[i].sum()-verbs_df.at[verbs_index[i],verbs_index[i]]
    tn = verbs_df.at[verbs_index[i],verbs_index[i]]-verbs_df[verbs_index[i]].sum()-verbs_df.iloc[i].sum()
    for n in range(len(verbs_index)):
        tn += verbs_df.iloc[n].sum()
    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    fscore = 2*precision*recall/(precision+recall)
    print(verbs_index[i], round(precision, 3), round(recall, 3), round(fscore, 3))


