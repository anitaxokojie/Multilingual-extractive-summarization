# Dataset Information

## TED Talk Dataset

This project uses the TED Talks dataset available on Kaggle:
[TED Talks dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset)

### Dataset Description

The dataset contains transcripts of TED Talks in multiple languages, including English and Spanish. For this project, I worked with a subset of 1,000 TED Talk transcripts that were available in both languages.

### Data Structure

The original dataset includes the following columns:
- talk_id: Unique identifier for each talk
- title: Title of the talk
- speaker_1: Main speaker name
- speakers: All speakers in the talk
- occupations: Speaker occupations
- about_speakers: Information about the speakers
- views: Number of views
- recorded_date: Date when the talk was recorded
- published_date: Date when the talk was published
- event: The TED event where the talk was presented
- native_lang: Original language of the talk
- available_lang: Languages in which the talk is available
- comments: Number of comments
- duration: Talk duration in seconds
- topics: Topics covered in the talk
- related_talks: Related TED talks
- url: URL to the talk
- description: Short description of the talk
- transcript: Full transcript of the talk

For our project, we primarily used these fields:
- talk_id
- title
- transcript

### Preprocessing

The data was preprocessed by:
1. Cleaning special characters and normalizing spaces
2. Tokenizing the text into sentences
3. Removing stopwords
4. Lemmatizing tokens

Language-specific preprocessing was applied for English and Spanish using the respective spaCy models.

### Sample Data

Due to the size of the dataset, I do not include the raw data in this repository. To reproduce my results, please download the dataset from the Kaggle link above.

### Citation

If you use this dataset in your research, please cite: Miguel Corral Jr. (2020). "TED Ultimate Dataset." Kaggle. 
Available at: https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset

