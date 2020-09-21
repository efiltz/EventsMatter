# EventsMatter

This repository contains a sample code to extract events and their components from court decisions. 
Due to the size of the models we do not include the language models but refer to the used framework Flair (https://github.com/flairNLP/flair) which is used to train/finetune language models used for this project.

 1. Get the corpus and split it into 5 splits
 2. Generate for each file a csv (to extract events) and a conll file (event components)
 3. finetune LM with the corpus
 4. run scripts
