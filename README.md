# cdc-vaers-llm
VAERS – Vaccine Adverse Event Reporting System 
1.	Main site - VAERS - Data Sets (hhs.gov)
2.	Data use - DEPARTMENT OF HEALTH AND HUMAN SERVICES (hhs.gov)
3.	Data collection – passive collection of online or via fillable pdf reports
4.	MedDRA symptoms
    a.	The MedDRA codes provided in the dataset are called the "Preferred Terms"; there are more than 17,000 Preferred Term codes
    b.	2023 1st symptom column has 136143 codes, 4594 unique 
    c.	Code #18 and #19, the description of the adverse event using MedDRA dictionary
        i.	The fields described in this table provide the adverse event coded terms utilizing the MedDRA dictionary. Coders will search for specific terms in Items 18 and 19 in VAERS 2 form or Boxes 7 and 12 on the VAERS 1 form and code them to a searchable and consistent MedDRA term; note that terms are included in the .csv file in alphabetical order. 
        ii.	There can be an unlimited amount of coded terms for a given event. Each row in the .csv will contain up to 5 MedDRA terms per VAERS ID; thus, there could be multiple rows per VAERS ID. For each of the VAERS_ID’s listed in the VAERSDATA.CSV table, there is a matching record in this file, identified by VAERS_ID.

