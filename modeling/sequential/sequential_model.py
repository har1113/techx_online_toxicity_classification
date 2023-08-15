#!/usr/bin/env python
# coding: utf-8

# In[73]:


from sklearn.base import BaseEstimator, TransformerMixin

class SequentialModel(BaseEstimator, TransformerMixin):
    
    def __init__(self, binary_model, multilabel_model):
        self.binary_model = binary_model
        self.multilabel_model = multilabel_model
        
    
    def transform(self, X, y=None):
        binary_predictions = self.binary_model.predict(X)
        filtered_indices = [i for i, pred in enumerate(binary_predictions) if pred == 1]
        filtered_data = X[filtered_indices]
        multilabel_predictions = self.multilabel_model.predict(filtered_data)
        return multilabel_predictions
    
    def predict(self, text):
        binary_prediction = self.binary_model.predict([text])
        if binary_prediction[0] == 1:
            multilabel_prediction = self.multilabel_model.predict([text])
            predicted_labels = multilabel_prediction.toarray().flatten()
            toxicity_types = ['toxic', 'severe_toxic', 'obscene', 
           'threat', 'insult', 'identity_hate']
            predicted_column_names = [toxicity_types[i] for i, label in enumerate(predicted_labels) if label == 1]
            return predicted_column_names
        else:
            return ['neutral']

  


# In[ ]:





# In[ ]:




