# Part-of-Speech (POS) Tagger using Hidden Markov and Viterbi Algorithm

## Project overview

This script implements a **Part-of-Speech (POS) tagger** using unigram and bigram models with the Viterbi algorithm. It calculates transition and emission probabilities from training datasets, applies the Viterbi algorithm to untagged test sentences, and outputs tagged sentences.

This project focuses on developing a part-of-speech (POS) tagger for the Irish language to investigate the syntactical changes in the language over time. The aim is to help inform language policy and investment decisions by analysing the written Irish language across a large time span, providing insights into the impact of English influence on Irish grammar.

The main objectives of this project are:
- **Develop a POS Tagger:** Automatically label tokens in Irish sentences with their grammatical function.
- **Train on Diachronic Corpus:** Train and test the POS tagger on a corpus of texts from 1960 to 2020.
- **Analyse Syntactical Changes:** Investigate changes in syntactical features like sentence complexity and verb forms over time.

HMM & Viterbi Algorithm:** Used for part-of-speech tagging and sequence prediction.

---

## Table of contents


- [Project overview](#project-overview) 
- [Usage instructions](#usage-instructions)
- [Performance metrics](#performance-metrics)
- [License](#license) 


---

## Usage instructions


TO RUN THIS POS TAGGER, PLACE YOUR TAGGED CORPUS IN THE SAME FOLDER AS THE JUPYTER NOTEBOOK.

THE TAGGED CORPUS MUST BE A CSV FILE. SAVE TWO COPIES OF THE CORPUS RENAMED AS FOLLOWS:
* trainingset_transmissions.csv
* trainingset_emissions.csv

DATA MUST BE SAVED IN THE FOLLOWING FORMAT:
<START>_START,token_tag,token_tag,token_tag,token_tag,<END>_END
(OPTIONAL: Lists of additional tagged tokens may be added to the emission dataset to avoid zero-probabilities of common words, such as verbs or numbers)

PLACE YOUR UNTAGGED TEXT IN THE SAME FOLDER AS THE JUPYTER NOTEBOOK.
RENAME THE FILE AS FOLLOWS:
* testset.csv

DATA MUST BE SAVED IN THE FOLLOWING FORMAT:
<START>,token,token,token,token,<END>




1. **Training Data**:
   - Two training files must be placed in the same folder as the script:
     - `trainingset_transmissions.csv`
     - `trainingset_emissions.csv`
   - **Format**:
     ```
     <START>_START,token1_tag,token2_tag,...,tokenN_tag,<END>_END
     ```

   - Optional: Add additional tagged tokens (e.g., verbs or numbers) in `trainingset_emissions.csv` to avoid zero probabilities for common tokens.

2. **Test Data**:
   - An untagged test file named `testset.csv` must be in the same folder.
   - **Format**:
     ```
     <START>,token1,token2,...,tokenN,<END>
     ```

---

## Outputs

- **Tagged Sentences**:
  - `testset_tagged_with_unigrams.csv`: Tagged sentences using the unigram model.
  - `testset_tagged_with_bigrams.csv`: Tagged sentences using the bigram model.

- **Data Tables**:
  - Transition and emission probabilities are printed as dataframes for debugging or visualization.

---

## How to Run

1. **Prepare Training and Test Data**:
   - Place the files `trainingset_transmissions.csv`, `trainingset_emissions.csv`, and `testset.csv` in the same directory as the script.

2. **Run the Script**:
   - Use Python to execute the script. It will:
     - Parse training data to calculate transition and emission probabilities.
     - Apply the Viterbi algorithm to tag the test dataset.

3. **Check Outputs**:
   - The tagged sentences are saved in the specified CSV files.

---

## Implementation Details

### 1. Training Data Parsing

- Parses training datasets (`trainingset_transmissions.csv` and `trainingset_emissions.csv`) into:
  - **Transition Probabilities**:
    - Unigram transitions: Probability of a tag being followed by another tag.
    - Bigram transitions: Probability of a tag pair being followed by another tag.
  - **Emission Probabilities**:
    - Probability of a word being tagged with a specific POS.

### 2. Test Data Parsing

- Parses untagged test sentences from `testset.csv`.



---

## Performance metrics
- **Accuracy:** Overall accuracy and accuracy per decade.
- **Precision, Recall, and F-Score:** Evaluated for individual verb tags, complexity markers, and the overall model.
- Analysis was conducted using both unigram and bigram transmission probabilities.


---

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE.txt) file for details.

Copyright (c) [2024] [Daniel White]

