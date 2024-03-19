# Incident Response Fairness Analysis

This project focuses on analyzing the fairness of incident response processes within an IT company. The analysis is based on an event log extracted from the audit system of an instance of the ServiceNow™ platform used by the organization.

## Dataset

The dataset provides insights into the incident management process and includes the following details:

- Number of instances: 141,712 events (24,918 incidents)
- Number of attributes: 36 attributes
  - 1 case identifier
  - 1 state identifier
  - 32 descriptive attributes
  - 2 dependent variables

The data was anonymized for privacy, and information from a relational database was used to enrich the event log.

## Data Cleaning

The data cleaning process involved several steps, including:

1. Handling missing values in date columns
2. Dropping columns with many "?" values
3. Removing rows with "?" values in specific columns
4. Extracting numeric parts from text columns
5. Filtering rows based on the 'incident_state' attribute

## Exploratory Data Analysis

The exploratory data analysis phase involved visualizing various aspects of the dataset, such as incident priorities, impact, urgency, and resolution times.

## Fairness Analysis

The fairness analysis was conducted using the Chi-Square test, which is a statistical method used to determine if there is a significant association between categorical variables. The analysis revealed significant associations between the 'assigned_to' attribute and various factors, such as reassignment count, knowledge, priority confirmation, and resolution time. These findings suggest potential unfairness in the assignment or decision-making processes, highlighting areas for further investigation and improvement.

## Machine Learning

The project also involved a machine learning component to further explore the fairness of incident handling. Key steps included:

1. Calculating mean values for specific attributes ('assigned_to', 'reassignment_count', and 'resolution_time')
2. Selecting relevant columns for fairness analysis
3. Defining fairness thresholds based on mean values
4. Creating a binary target variable 'fairness' based on the defined criteria
5. Preprocessing the data using a ColumnTransformer for one-hot encoding and preserving numerical features
6. Training a RandomForestClassifier model on the training data
7. Evaluating the model's performance on the test data

The trained model achieved an accuracy of 86.67% on the test dataset.

## Streamlit App Demo


https://github.com/MohsenMaaleki/Incident-Response-Fairness-Analysis/assets/95247257/e8d0e662-6ae1-4b2f-8351-d989f023775c



