# Usage

Get your logits ready in the format of 
((y_logits_val, y_val), (y_logits_test, y_test)). If you are not sure of what 
the data format should be, please check the `sample_logits.p `file with 
`unpickle_logits` function.

To calibrate your model, simply run `model_calibration` function within 
`scaling_diagrams.py`. Logit files should be kept in a list so it is easier for comparision