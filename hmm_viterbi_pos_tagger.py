

# Import libraries
import pandas as pd

# OPEN CSV FILE FOR TRANSMISSION PROBABILITIES
df1 = pd.read_csv('trainingset_transmission.csv', low_memory=False)
single_str_list = df1.values.tolist()


# Convert the training set into tuples of tokens and tags
table = []
for sent in single_str_list:
    table.append(sent[1:])


#Split tags back into list of lists (sentences) of tuples:
tuple_list = []
for sent in table:
    new_sent = []
    for tag_pair in sent:
        if isinstance(tag_pair, str):
            for i in range(len(tag_pair)):
                if tag_pair[i] == '_':
                    new_sent.append((tag_pair[:i], tag_pair[i+1:]))
    tuple_list.append(new_sent)

print(tuple_list[-1:])


#### CREATE TABLE OF Transition probabilities for 1-grams ###
# Create a dataframe with row indexes and column indexes both being the TAGS
# Also iterate through and count how many times each TAG (row) is followed by other TAG

# For row indexes, create list of tags used in the corpus
row_indexes = []
for sent in tuple_list:
    for tup in sent:
        if tup[1] not in row_indexes:
            row_indexes.append(tup[1])

#Create dict with a key for each column in the table, and all values set to zero
empty_unigram_dict = {}
for tag in row_indexes:
    empty_unigram_dict[tag] = 0

#Create a list of dicts, where each dict is one row of the table
unigram_count = []
for tag in row_indexes:
    unigram_count.append(empty_unigram_dict.copy())
    

# Iterate through corpus to count frequency of each tag following each tag
for n in range(len(row_indexes)):
    for sent in tuple_list:
        for i in range(len(sent)-1):
            if sent[i][1] == row_indexes[n]:
                unigram_count[n][sent[i+1][1]]+=1


#Create copy of the table, then calculate averages by dividing each cell in row by the row_total
unigram_data = []
for item in unigram_count:
    unigram_data.append(item)

for n in range(len(unigram_count)):
    row_total = 0
    for key, value in unigram_count[n].items():
        row_total+= value
    if row_total!= 0:
        for k, v in unigram_data[n].items():
            unigram_data[n][k]=v/row_total




# Save data as dataframe
unigrams_df = pd.DataFrame(unigram_data, row_indexes)
print(unigrams_df)


#### CREATE TABLE OF Transition probabilities for 2-grams ###
# Create a dataframe with row indexes being 2-grams of TAGS and column indexes being the following TAG
# Iterate through and count how many times each 2-gram of TAGs (row) is followed by each other TAG


bi_row_indexes = []
for sent in tuple_list:
    for i in range(len(sent)-1):
        if (sent[i][1], sent[i+1][1]) not in bi_row_indexes:
            bi_row_indexes.append((sent[i][1], sent[i+1][1]))

# Create row index labels by contcatenating the tuples
bi_row_indexes_concat = []
for item in bi_row_indexes:
    bi_row_indexes_concat.append("_".join(item))



#Create dict with a key for each column in the table, and all values set to zero
empty_bigram_dict = {}
for tag in row_indexes:
    empty_bigram_dict[tag] = 0

#Create a list of dicts, where each dict is one row of the table
bigram_count = []
for tag in bi_row_indexes_concat:
    bigram_count.append(empty_bigram_dict.copy())
    


# Iterate through corpus to count frequency of each tag following each tag
for n in range(len(bi_row_indexes_concat)):
    for sent in tuple_list:
        for i in range(len(sent)-2):
            if "_".join((sent[i][1], sent[i+1][1])) == bi_row_indexes_concat[n]:
                bigram_count[n][sent[i+2][1]]+=1
    
#Create copy of the table, then calculate averages by dividing each cell in row by the row_total
bigram_data = []
for item in bigram_count:
    bigram_data.append(item)

for n in range(len(bigram_count)):
    bi_row_total = 0
    for key, value in bigram_count[n].items():
        bi_row_total+= value
    if bi_row_total!= 0:
        for k, v in bigram_data[n].items():
            bigram_data[n][k]=v/bi_row_total


# Save data as dataframe
bigrams_df = pd.DataFrame(bigram_data, bi_row_indexes_concat)
print(bigrams_df)





# OPEN CSV FILE FOR EMISSION PROBABILITIES
df2 = pd.read_csv('trainingset_emission.csv', low_memory=False)
single_str_list2 = df2.values.tolist()

table2 = []
for sent in single_str_list2:
    table2.append(sent[1:])


#Split tags back into list of lists (sentences) of tuples:
tuple_list2 = []
for sent in table2:
    new_sent = []
    for tag_pair in sent:
        tag_pair = str(tag_pair)
        for i in range(len(tag_pair)):
            if tag_pair[i] == '_':
                new_sent.append((tag_pair[:i], tag_pair[i+1:]))
    tuple_list2.append(new_sent)

print(tuple_list2[-1:])




#### CREATE TABLE OF emission probabilities ###

# Create a list of all tokens in corpus
tokens_list = []
for sent in tuple_list2:
    for tup in sent:
        if tup[0] not in tokens_list:
            tokens_list.append(tup[0])



#Create dict with a key for each column in the table, and all values set to zero
empty_token_dict = {}
for tok in tokens_list:
    empty_token_dict[tok] = 0

#Create a list of dicts, where each dict is one row of the table
token_count = []
for tag in row_indexes:
    token_count.append(empty_token_dict.copy())
    


# Iterate through corpus to count frequency of each token assigned as each tag
for n in range(len(row_indexes)):
    for sent in tuple_list:
        for i in range(len(sent)):
            if sent[i][1] == row_indexes[n]:
                token_count[n][sent[i][0]]+=1
    
    
#Create copy of the table, then calculate averages by dividing each cell in row by the row_total
emission_data = []
for item in token_count:
    emission_data.append(item)

for n in range(len(token_count)):
    row_total = 0
    for key, value in token_count[n].items():
        row_total+= value
    if row_total!= 0:
        for k, v in emission_data[n].items():
            emission_data[n][k]=v/row_total


# Save data as dataframe
emission_df = pd.DataFrame(emission_data, row_indexes)
print(emission_df)


# IMPORT TESTSET TO TAG
df3 = pd.read_csv('testset.csv', low_memory=False)
single_str_list3 = df3.values.tolist()

index_removed = []
for sent in single_str_list3:
    index_removed.append(sent[1:])

test_dataset = []
for sent in index_removed:
    new_sent = []
    for i in range(len(sent)):
        if sent[i] == '<END>':
            test_dataset.append(sent[:i+1])

print(test_dataset[0])



#### Viterbi algorithm for uni-grams

# First find the possible tags for each token with their non-zero emission probability
tagged_sents_unigrams_tuples = []
for sent in test_dataset:
    # Initialise value start tag
    pathway_values = {'START':1.0}
    pathway_labels = {'START':'START'}
    for n in range(len(sent)-1):
        poss_token_tags = {}
        # Resolve OOV tokens
        if sent[n+1] not in emission_df.columns:
            # Assume capitalised words are proper nouns
            if len(sent[n+1]) > 1 and ((n > 0 and sent[n+1].isalpha() and sent[n+1][0].isupper() and len(sent[n+1]) > 1) or (n > 0 and sent[n+1].isalpha() and sent[n+1][1].isupper() and len(sent[n+1]) > 1)):
                poss_token_tags = {'prop':1.0}
            # Otherwise, use the top three transmission probabilities from the previous token to define placeholder tags
            else:
                for m in pathway_values.keys():
                    values_top3_trans = unigrams_df.loc[m].nlargest(3).values.tolist()
                    index_top3_trans = unigrams_df.loc[m].nlargest(3).index.values.tolist()
                    for k in range(len(values_top3_trans)):
                        # Exclude any zero-probabilties
                        if values_top3_trans[k] != 0.0:
                            # Add tags and their highest probabilities to dictionary for current token
                            if index_top3_trans[k] not in poss_token_tags.keys():
                                poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                            elif values_top3_trans[k] > poss_token_tags[index_top3_trans[k]]:
                                poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                        else:
                            poss_token_tags = {'for':1.0}
        else:
            # For each known token, find emmision probaility from table
            tag_index = emission_df[sent[n+1]].index.values.tolist()
            tag_column = emission_df[sent[n+1]].values.tolist()
            for t in range(len(tag_index)):
                # Exclude emmissions with zero-probability
                if tag_column[t]>0:
                    for m in pathway_values.keys():
                        if tag_index[t] not in poss_token_tags.keys():
                            poss_token_tags[tag_index[t]] = tag_column[t]
                        elif tag_column[t] > poss_token_tags[tag_index[t]]:
                            poss_token_tags[tag_index[t]] = tag_column[t]
        # Update the dictionary of the most probable tag pathways
        temp_pathway_values = {}
        temp_pathway_labels = {}   
        for j in poss_token_tags.keys():
            highest_path_to_j = 0
            label_of_jtoken = ''
            for i in pathway_values.keys():
                if unigrams_df.at[i,j] > 0.0:
                    if len(sent) > 20:
                        current_path = pathway_values[i]*unigrams_df.at[i,j]*poss_token_tags[j]*100
                    else:
                        current_path = pathway_values[i]*unigrams_df.at[i,j]*poss_token_tags[j]*10
                    if current_path > highest_path_to_j:
                        highest_path_to_j = current_path
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[j] = highest_path_to_j
                elif not unigrams_df.at[i,j] and len(poss_token_tags) == 1:
                    if len(sent) > 20:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*100
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[j] = highest_path_to_j
                    else:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*10
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[j] = highest_path_to_j
        # Update the dictionary of the most probable tag pathways
        pathway_values = temp_pathway_values
        pathway_labels = temp_pathway_labels
    # Append the final most probable tag pathway to the list of sentences
    final_tag_list = []
    tagged_sentence = []
    for v in pathway_labels.values():
        marker = -1
        for i in range(len(v)):
            marker+=1
            if v[i] == '_':
                final_tag_list.append(v[i-marker:i])
                marker = -1
        final_tag_list.append(v[i-marker:])
        for n in range(len(sent)):
            tagged_sentence.append((sent[n], final_tag_list[n]))
    tagged_sents_unigrams_tuples.append(tagged_sentence)



print(len(tagged_sents_unigrams_tuples), 'sentences tagged with unigrams.')




#### Viterbi algorithm for bi-grams

# First find the possible tags for each token with their non-zero emission probability
tagged_sents_bigrams_tuples = []
for sent in test_dataset:
# RUNS THE UNIGRAM MODEL FIRST TO GET PATHWAY FOR FIRST TWO TAGS
    # Initialise value start tag
    count = 0
    pathway_values = {'START':1.0}
    pathway_labels = {'START':'START'}
    for n in range(1):
        poss_token_tags = {}
        # Resolve OOV tokens
        if sent[n+1] not in emission_df.columns:
            # Assume capitalised words are proper nouns
            if len(sent[n+1]) > 1 and ((n > 0 and sent[n+1].isalpha() and sent[n+1][0].isupper() and len(sent[n+1]) > 1) or (n > 0 and sent[n+1].isalpha() and sent[n+1][1].isupper() and len(sent[n+1]) > 1)):
                poss_token_tags = {'prop':1.0}
            # Otherwise, use the top three transmission probabilities from the previous token to define placeholder tags
            else:
                for m in pathway_values.keys():
                    # Find the three top values of unigram transmissions from table and their indices for each possible tag of the previous token
                    values_top3_trans = unigrams_df.loc[m].nlargest(3).values.tolist()
                    index_top3_trans = unigrams_df.loc[m].nlargest(3).index.values.tolist()
                    for k in range(len(values_top3_trans)):
                        # Exclude any zero-probabilties
                        if values_top3_trans[k] != 0.0:
                            # Add tags and their highest probabilities to dictionary for current token
                            if index_top3_trans[k] not in poss_token_tags.keys():
                                poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                            elif values_top3_trans[k] > poss_token_tags[index_top3_trans[k]]:
                                poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                        else:
                            poss_token_tags = {'for':1.0}
        else:
            # For each known token, find emmision probaility from table
            tag_index = emission_df[sent[n+1]].index.values.tolist()
            tag_column = emission_df[sent[n+1]].values.tolist()
            for t in range(len(tag_index)):
                # Exclude emmissions with zero-probability
                if tag_column[t]>0:
                    for m in pathway_values.keys():
                        if tag_index[t] not in poss_token_tags.keys():
                            poss_token_tags[tag_index[t]] = tag_column[t]
                        elif tag_column[t] > poss_token_tags[tag_index[t]]:
                            poss_token_tags[tag_index[t]] = tag_column[t]
        # Find the most probable pathway to each node
        temp_pathway_values = {}
        temp_pathway_labels = {}   
        for j in poss_token_tags.keys():
            highest_path_to_j = 0
            label_of_jtoken = ''
            for i in pathway_values.keys():
                if unigrams_df.at[i,j] > 0.0:# Find the most probable pathway to each node
                    current_path = pathway_values[i]*unigrams_df.at[i,j]*poss_token_tags[j]
                    if current_path > highest_path_to_j:
                        highest_path_to_j = current_path
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[j] = highest_path_to_j
                elif not unigrams_df.at[i,j] and len(poss_token_tags) == 1:
                    highest_path_to_j = pathway_values[i]*poss_token_tags[j]
                    label_of_jtoken = '_'+j
                    temp_pathway_labels[j] = pathway_labels[i]+label_of_jtoken
                    temp_pathway_values[j] = highest_path_to_j
        pathway_values = temp_pathway_values
        pathway_labels = temp_pathway_labels
        count+=1
    if count == 1:
        if len(pathway_labels) > 0:
            pathway_values['START'+'_'+max(pathway_labels)] = 1.0
            pathway_labels['START'+'_'+max(pathway_labels)] = 'START'+'_'+max(pathway_labels)
        else:
            pathway_values['START_pun'] = 1.0
            pathway_labels['START_pun'] = 'START_pun'


    
    # PART II: BIGRAM MODEL FOR TOKENS n+3 ONWARDS
    for n in range(len(sent)-2):
        poss_token_tags = {}
        # Resolve OOV tokens
        if sent[n+2] not in emission_df.columns:
            # Assume capitalised words are proper nouns
            if len(sent[n+2]) > 1 and (n > 0 and sent[n+2].isalpha() and sent[n+2][0].isupper() and len(sent[n+2]) > 1 or n > 0 and sent[n+2].isalpha() and sent[n+2][1].isupper() and len(sent[n+2]) > 1):
                poss_token_tags = {'prop':1.0}
            # If not proper noun, use the top three transmission probabilities from the previous token to define placeholder tags
            else:
                for m in pathway_values.keys():
                    if m in bigrams_df.index:
                        # Find the three top values of unigram transmissions from table and their indices for each possible tag of the previous token
                        values_top3_trans = bigrams_df.loc[m].nlargest(3).values.tolist()
                        index_top3_trans = bigrams_df.loc[m].nlargest(3).index.values.tolist()
                        for k in range(len(values_top3_trans)):
                            # Exclude any zero-probabilties
                            if values_top3_trans[k] != 0.0:
                                # Add tags and their highest probabilities to dictionary for current token
                                if index_top3_trans[k] not in poss_token_tags.keys():
                                    poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                                elif values_top3_trans[k] > poss_token_tags[index_top3_trans[k]]:
                                    poss_token_tags[index_top3_trans[k]] = values_top3_trans[k]*pathway_values[m]
                            else:
                                poss_token_tags = {'for':1.0}
                    else:
                        for c in row_indexes:
                            poss_token_tags[c] = float(1/len(row_indexes))
        else:
            # For each known token, find emmision probaility from table
            tag_index = emission_df[sent[n+2]].index.values.tolist()
            tag_column = emission_df[sent[n+2]].values.tolist()
            for t in range(len(tag_index)):
                # Exclude emmissions with zero-probability
                if tag_column[t]>0:
                    for m in pathway_values.keys():
                        if tag_index[t] not in poss_token_tags.keys():
                            poss_token_tags[tag_index[t]] = tag_column[t]
                        elif tag_column[t] > poss_token_tags[tag_index[t]]:
                            poss_token_tags[tag_index[t]] = tag_column[t]
        # Update the dictionary of the most probable tag pathways
        temp_pathway_values = {}
        temp_pathway_labels = {}   
        for j in poss_token_tags.keys():
            highest_path_to_j = 0
            label_of_jtoken = ''
            for i in pathway_values.keys():
                old_i = ''
                for letter_num in range(len(i)):
                    if i[letter_num] == '_':
                        old_i = i[letter_num+1:]
                if i in bigrams_df.index and bigrams_df.at[i,j] > 0.0:
                    if len(sent) > 20:
                        current_path = pathway_values[i]*bigrams_df.at[i,j]*poss_token_tags[j]*100
                    else:
                        current_path = pathway_values[i]*bigrams_df.at[i,j]*poss_token_tags[j]*10
                    if current_path > highest_path_to_j:
                        highest_path_to_j = current_path
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[old_i+'_'+j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[old_i+'_'+j] = highest_path_to_j
                elif i in bigrams_df.index and bigrams_df.at[i,j] == 0.0 and len(poss_token_tags) == 1:
                    if len(sent) > 20:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*100
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[old_i+'_'+j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[old_i+'_'+j] = highest_path_to_j
                    else:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*10
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[old_i+'_'+j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[old_i+'_'+j] = highest_path_to_j
                else:
                    if len(sent) > 20:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*100
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[old_i+'_'+j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[old_i+'_'+j] = highest_path_to_j
                    else:
                        highest_path_to_j = pathway_values[i]*poss_token_tags[j]*10
                        label_of_jtoken = '_'+j
                        temp_pathway_labels[old_i+'_'+j] = pathway_labels[i]+label_of_jtoken
                        temp_pathway_values[old_i+'_'+j] = highest_path_to_j
        pathway_values = temp_pathway_values
        pathway_labels = temp_pathway_labels
    results = {}
    for r in pathway_values.keys():
        results[pathway_labels[r]] = pathway_values[r]
    if len(results) > 0:
        max_results = max(results)
        final_tag_list = []
        tagged_sentence = []
        marker = -1
        for i in range(len(max_results)):
            marker+=1
            if max_results[i] == '_':
                final_tag_list.append(max_results[i-marker:i])
                marker = -1
        final_tag_list.append(max_results[i-marker:])
        for n in range(len(sent)):
            tagged_sentence.append((sent[n], final_tag_list[n]))
        tagged_sents_bigrams_tuples.append(tagged_sentence)

print(len(tagged_sents_bigrams_tuples), 'sentences tagged with bigrams.')





# COVERT TUPLES TO LIST OF STRINGS AND SAVE AS CSV
new_list = []
for sent in tagged_sents_unigrams_tuples:
    new_sent = []
    for tup in sent:
        new_sent.append("_".join(tup))
    new_list.append(new_sent)
df_final = pd.DataFrame(new_list)
df_final.to_csv("testset_tagged_with_unigrams.csv")


new_list = []
for sent in tagged_sents_bigrams_tuples:
    new_sent = []
    for tup in sent:
        new_sent.append("_".join(tup))
    new_list.append(new_sent)
df_final = pd.DataFrame(new_list)
df_final.to_csv("testset_tagged_with_bigrams.csv")

print('Files saved.')