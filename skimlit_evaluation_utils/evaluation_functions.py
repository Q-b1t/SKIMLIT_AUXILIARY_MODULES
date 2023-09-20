from sklearn.metrics import accuracy_score,precision_recall_fscore_support # keep this import the first one (https://stackoverflow.com/questions/67735216/after-using-pip-i-get-the-error-scikit-learn-has-not-been-built-correctly)

def calculate_results(y_true,y_pred):
  # calculate model accuracy
  model_accuracy = accuracy_score(y_true,y_pred) * 100
  # calculate the model's presicion, recall, and f1score using a "weighted" average
  model_presicion, model_recall,model_f1,_ = precision_recall_fscore_support(y_true,y_pred,average="weighted")
  model_results = {
      "accuracy":model_accuracy,
      "precision":model_presicion,
      "recall":model_recall,
      "f1":model_f1
  }
  return model_results

