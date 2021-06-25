# CustomModel
## Custom Detection model trained on SVHN Dataset to Recognise House

#Steps:

pip install -r requirement.txt

Download Data from below link

train.tar.gz http://ufldl.stanford.edu/housenumbers/train.tar.gz

test.tar.gz http://ufldl.stanford.edu/housenumbers/test.tar.gz

Exract Files

Convert mat file to csv file for easy data preprocessing. Use Data-prep notebook for that.

Start Training 
    
    python train.py

For Inference

    python predict.py


#Google Colab used for training,
Link: https://colab.research.google.com/drive/1WoTTFc8yNqTW85b7Cfla1g93DmFAuN3N?usp=sharing
