# Problem description

A [user survey](https://activeclean.github.io/files/hilda-2016.pdf) found that 
data cleaning is not a simple process that can be easily accomplished by individuals. 

It is often iterative rather than sequential, requesting multiple reviews.

Additionally, data quality evaluation is not easy as there is often no clear 
procedure. 

These challenges are compounded as datasets get larger, 
the cleaning process becomes even more complex.

Impact of poor cleaning on model accuracy is illustrated here:
![Image showing impact of poor cleaning, 
which can be worse in skewing the model then not cleaning](Images/ImpactOfPoorCleaning.png)

Source: [ActiveClean Website](https://activeclean.github.io/)

# Solution: ActiveClean

To reduce data-cleaning mistakes, ActiveClean takes humans out of the two most error-prone steps
of data cleaning using Machine Learning: 
* Finding dirty data
* Updating the model. 

To see how well it worked, researchers compared the tool’s results against two 
baseline methods using ProPublica’s Dollars for Docs database of 240,000 records on corporate donations to doctors
* A model that was retrained based on an edited subset of the data
* A prioritization algorithm called Active Learning that chooses informative labels for unclear data.

Significant Result:
* Without data cleaning, a model of the dataset could predict an improper donation 
66 percent of the time
* ActiveClean raised that rate to 90 percent after cleaning only 5,000 records, 
while the Active Learning method had to be applied to 50,000 records to achieve the same rate.


# How can we build onto the existing research?

We can first make the codebase for ActiveClean more accessible by making a Python package.

Each segment of the ActiveClean process will be a submodule.