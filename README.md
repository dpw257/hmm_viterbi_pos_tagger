# Part-of-Speech (POS) Tagger using Hidden Markov and Viterbi Algorithm

## Project overview

This project develops a **Part-of-Speech (POS) tagger** using a Hidden Markov model with a Viterbi algorithm. It calculates emission and transmission probabilities between unigram and bigram tokens from the training data to tag a diachronic corpus of untagged test sentences from the years 1960 to 2020. This aim of the project is to investigate the syntactical changes in the Irish language over time. 

The main objectives of this project are:
- **Develop a POS Tagger:** Automatically label tokens in Irish sentences with their grammatical function.
- **Train on Diachronic Corpus:** Train and test the POS tagger on a corpus of texts from 1960 to 2020.
- **Analyse Syntactical Changes:** Investigate changes in two syntactical features (sentence complexity and verb forms).

---

## Table of contents


- [Project overview](#project-overview) 
- [Usage instructions](#usage-instructions)
- [Performance metrics](#performance-metrics)
- [License](#license) 


---

## Usage instructions
1. **Format of datasets**

- **Preparing the training data**
The training datasets are a corpus of tagged sentences to calculate emission and transmission probabilities:
	- trainingset_transmissions.csv
	- trainingset_emissions.csv

The tagged sentences in the `.csv` training data files must be in this format:
```
<START>_START,token_tag,token_tag,token_tag,...token_tag,<END>_END
```

OPTIONAL: Lists of additional tagged tokens may be added to the emission dataset to avoid zero-probabilities of common words, such as verbs or numbers)
(Optional: Add lists of additional tagged tokens (e.g. numbers, common English words, common verb forms) to the file `trainingset_emissions.csv` to lower the frequency of out of vocabulary tokens (zero-emission probabilities.)

- **Testing data**
The test data is a corpus of unseen sentences (25% of the original corpus). Two versions are needed; an untagged version for the model to tag and a manually tagged version to evaluate the success of the model:
	- `testset.csv`
	- `testset_answers.csv`

The untagged unseen sentences in the `.csv` file must be in this format:
```
<START>,token,token,token,token,<END>
```

The tagged answer sentences must be in this format:
```
<START>_START,token_tag,token_tag,token_tag,...token_tag,<END>_END
```
 
2. **Tagging the unseen data**:
- Place the files `trainingset_transmissions.csv`, `trainingset_emissions.csv`, and `testset.csv` in the same directory as the `.py` file.
- Open the Python code `hmm_viterbi_pos_tagger.py` and check that the paths to the training sets are correct. 
- Check the paths for saving the output `.csv` files is named correctly (i.e. name includes publication dates of corpus texts and unigram/bigram)
- Run the code to:
	- Calculate the transition and emission probabilities from your tagged corpus.
	- Apply the HMM/Viterbi algorithm to tag the test dataset for both unigram and bigram transmissions
- The tagged sentences are saved in the specified CSV files.
- Run the code again for any further data files (i.e. corpus files from different time periods).

3. **Evaluating the model**
- Open the Python script `hmm_evaluation.py` and check the variables `decades` (the list of decades used in your corpus) and `file_types` matches the names of your `.csv` files. Three files per time period are needed:
	- `testset_1960_answers.csv`
	- `testset_1960_bigrams_tagged.csv`
	- `testset_1960_unigrams_tagged.csv`
- Run the code to output performance metrics for the model.





---

## Performance metrics
- All analyses were conducted for both **unigram and bigram** transmission probabilities.
- **Accuracy** calculated per decade and for the model overall.
- **Precision, Recall and F-Score** of individual verb tags, individual syntax tags and the model overall.
- **Confusion matrices** for syntax marker tags and verb form tags.

---

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE.txt) file for details.

Copyright (c) [2024] [Daniel White]

