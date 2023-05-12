The main code body of my function takes in the various arguments as specified in the assignment. You can run the file for sections 1,2 and 5 as mentioned in the assignment.
For example, if running the section 1(linear regression) - 
python main.py --train_path="train.csv" --val_path="validation.csv" --test_path="test.csv" --out_path="out.csv" --section=1

This would create a out.csv file in the location specified. This would contain the predicted values on the test data with their sample numbers mentioned.
<sample name 1>, <output score 1>
.
.
.
<sample name N>, <output score N>

For implementing other sections I have made specific functions to support other arguments like 3 and 4.
For the visualisation part, a lot of code is present within sections 1,2 and 5 itself commented so as to not cause any interference with the actual output of the program.
You can uncomment it to get further results.