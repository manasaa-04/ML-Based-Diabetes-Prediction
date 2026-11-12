This UTIEntryModel2017to2018readme.txt file was generated on 2020-10-13 by Jens Kjølseth Møller


GENERAL INFORMATION

1. Title of Dataset: UTIEntryModel2017to2018Dataset

2. Author Information
	A. Principal Investigator Contact Information
		Name: jens Kjølseth Møller
		Institution: Lillebaelt Hospital, University Hospital of Southern Denmark 
		Address: Beriderbakken 4, DK-7100 Vejle, Denmark
		Email: jens.kjoelseth.moeller@rsyd.dk

	B. Associate or Co-investigator Contact Information
		Name: Christian Hardahl
		Institution: SAS Institute A/S, Aarhus
		Address: Frederiks Plads 36, DK-8000 Aarhus C, Denmark 
		Email: Christian.Hardahl@sas.com

	C. Associate or Co-investigator Contact Information
		Name: Martin Sørensen
		Institution: SAS Institute A/S, Copenhagen
		Address: Købmagergade 7-9, DK-1150 København, Denmark
		Email: Martin.Soerensen@sas.com

3. Date of data collection: 2017-01-01 through 2018-04-30

4. Geographic location of data collection: Region of Southern Denmark 

5. Information about funding sources that supported the collection of the data: Region of Southern Denmark Research Foundation, grant no. 17/15659


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: No

2. Links to publications that cite or use the data: manuscript PONE-D-20-17945

3. Links to other publicly accessible locations of the data: No

4. Links/relationships to ancillary data sets: No

5. Was data derived from another source? No
	A. If yes, list source(s): 

6. Recommended citation for this dataset: Møller JK, Sørensen M, Hardahl C. Prediction of risk of acquiring urinary tract infection during hospital stay based on Machine-learning: A retrospective cohort study


DATA & FILE OVERVIEW

1. File List: 
UTIEntryModel2017to2018Dataset.xlsx (dataset in Microsoft Excel)
UTIEntryModel2017to2018readme.txt (ReadMe text file)


2. Relationship between files, if important: 

3. Additional related data collected that was not included in the current data package: 

4. Are there multiple versions of the dataset? No
	A. If yes, name of file(s) that was updated: 
		i. Why was the file updated? 
		ii. When was the file updated? 


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
Data are extracted from a regional data warehouse containing 
1) a copy of EMR notes from the four hospital trusts in the Region of Southern Denmark
2) administrative data on all hospital admissions 
3) positive culture results from the four clinical microbiology laboratories in the region 

2. Methods for processing the data: 
We developed models for UTI prediction with five machine-learning algorithms using demographic information, laboratory results, data on antibiotic treatment, 
past medical history (ICD10 codes), and clinical data by transformation of unstructured narrative text in Electronic Medical Records to structured data by Natural Language Processing.
Data were deidentified before submission and contain no sensitive or personally identifiable information. 
Patient_admission_id is just a sequential case number from 1 to 301932.

3. Instrument- or software-specific information needed to interpret the data: 
The tools used are SAS® Content Categorization (for text analytics), 
SAS® Data Integration Studio (for data integration/management), 
SAS® Enterprise Miner™ (for predictive modelling), 
and SAS® Visual Analytics (for operationalizing the results). 

4. Standards and calibration information, if appropriate: NA

5. Environmental/experimental conditions: NA

6. Describe any quality-assurance procedures performed on the data: 
All Danish residents have from cradle to grave the same unique civil registration number used for all health contacts in Denmark, 
which enables linkage between the various public healthcare registries and the construction of the medical history of a patient 

7. People involved with sample collection, processing, analysis and/or submission: 
All data collected from electronic health care registries; processing, analysis and submission performed by Principal Investigator and Co-Investigators.

DATA-SPECIFIC INFORMATION FOR: UTIEntryModel2017to2018Dataset.xlsx

1. Number of variables: 14

2. Number of cases/rows: 301932

3. Variable List: 
patient_admission_id, GENDER (M=Male;F=Female), age (in years), ADMISSION_TYPE (Acute or Planned), readmission (0 or 1), admission_hospital_text (Hospital trusts numbered 1-4), 
admitted_org_id_text (clinical departments numbered 1-135), ICD10_COPD (0 or 1), ICD10_urinary_retention (0 or 1), ICD10_neurological_disease (0 or 1), 
previous_HAI_UTI (number of), previous_CAI_UTI (number of), Previous_IUC (number of), TARGET_UTI_ENTRY (UTI present: 0 or 1), partition (train, validate, test); 0= absent, 1=present.

4. Missing data codes: blank=missing

5. Specialized formats or other abbreviations used: 
