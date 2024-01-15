# CDC VAERS Adverse Event Multi-Label Text Classification (Using DistilBERT)
This is a rapid research project to showcase my skills on getting up-to-speed on a project over the course of a single weekend.

## Background
The CDC and FDA collect vaccine adverse events through the Vaccine Adverse Event Reporting System (VAERS). There are 2 free text fields where descriptions of the events are captured. Coders search for specific terms in these fields and label them with searchable and consistent MedDRA terms. Using an LLM to code the text fields into the MedDRA terms would help automate the manual process and could lead to higher accuracy.

## Approach
A Fine-Tuned LLM DistilBERT to Classify User VARES Adverse Event Symptoms Text Descriptions To predict more than just the first symptom (SYMPTOM1

## Results
+ The base DistilBERT model was unable to label any symptoms correctly in the test set. (Base Model Accuracy 0%.)
+ By fine-tuning the model on just 200 records from early 2023, the model began to correctly label 10% of symptoms from a withheld test dataset. (Trained on 200 records in in 26 minutes, Fine-Tuned Model Accuracy:: 10%.)
+ By expanding to 20,000 records from early 2023, the model began to correctly label symptoms more often than not, labeling 55.5% of symptoms correctly. (Trained on 20,000 records in 252 minutes (4 hours), Fine-Tuned Model Accuracy: 55.6%.)

In conclusion, these results show that by training on just 19% (20,000/105,726) of a single year’s data, the 1st symptom column could already be predicted more often than not (55.6% accuracy). While developing a robust system requires additional work, these initial results demonstrate the exciting potential of investing additional time in a fine-tuned model expanded to all symptoms and more training records. 

## Next Steps
In the real-life scenario, we have the added complexity of needing multi-label text classification. These types of problems are typically done using a multi-label classification setup, where each symptom is treated as a separate label, and the model learns to predict the presence or absence of each symptom independently. This can also be done through fine-tuning DistilBERT. The main challenge is that this will take significantly more compute for fine-tune a larger labeling on more data records, so using a cloud service will be required.

## Appendix: VAERS – Vaccine Adverse Event Reporting System 
1.	Main site - https://vaers.hhs.gov/data/datasets.html
2.	Data use - https://vaers.hhs.gov/docs/VAERSDataUseGuide_November2020.pdf
3.	Data collection – passive collection of online or via fillable pdf reports 
4.	MedDRA symptoms coding
+	The MedDRA codes provided in the dataset are called the ''Preferred Terms''; there are more than 17,000 of these codes.
+	2023 1st symptom column has 136143 codes, 4594 of which are unique. 
+ Code #18 and #19, the description of the adverse event using MedDRA dictionary.
+  The fields described in this table provide the adverse event coded terms utilizing the MedDRA dictionary. Coders will search for specific terms in Items 18 and 19 in VAERS 2 form or Boxes 7 and 12 on the VAERS 1 form and code them to a searchable and consistent MedDRA term; note that terms are included in the .csv file in alphabetical order. 
+  There can be an unlimited amount of coded terms for a given event. Each row in the .csv will contain up to 5 MedDRA terms per VAERS ID; thus, there could be multiple rows per VAERS ID. For each of the VAERS_ID’s listed in the VAERSDATA.CSV table, there is a matching record in this file, identified by VAERS_ID.
5.	The pdf version of the form:
![image](https://github.com/drew6050/cdc-vaers-llm/assets/102396940/c61b1ca7-7822-44be-b0f8-a83b54a522a3)
