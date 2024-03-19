import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier



# read the dataset
def read_dataset():
    
    dataset_path = 'archive\incident_event_log.csv'
    
    try:
        # Attempt to read the dataset
        raw_data = pd.read_csv(dataset_path)
        return raw_data
    except Exception as e:
        st.error(f"Error reading the dataset: {e}")
        return None



def clean_dataset():

    raw_data = read_dataset()

    # List of date columns
    date_columns = ['opened_at', 'sys_created_at', 'sys_updated_at', 'resolved_at', 'closed_at']

    # List of columns to drop becouse they too many "?" value inside them
    columns_to_drop = ['problem_id', 'rfc', 'vendor', 'caused_by','cmdb_ci','sys_created_by']

    raw_data[date_columns] = raw_data[date_columns].apply(pd.to_datetime, errors='coerce')
    # Forward fill missing values in date columns
    raw_data[date_columns] = raw_data[date_columns].ffill()

    # clearning columns
    raw_data = raw_data[(raw_data['assignment_group'] != '?') & (raw_data['assigned_to'] != '?') & (raw_data['location'] != '?')
                        & (raw_data['caller_id'] != '?') & (raw_data['caller_id'] != '?')]

    # Extract numeric part  and convert to integer
    raw_data['opened_by'] = raw_data['opened_by'].str.extract('(\d+)').astype('Int64')
    raw_data['sys_updated_by'] = raw_data['sys_updated_by'].str.extract('(\d+)').astype('Int64')
    raw_data['location'] = raw_data['location'].str.extract('(\d+)').astype('Int64')
    raw_data['category'] = raw_data['category'].str.extract('(\d+)').astype('Int64')
    raw_data['subcategory'] = raw_data['subcategory'].str.extract('(\d+)').astype('Int64')
    raw_data['assignment_group'] = raw_data['assignment_group'].str.extract('(\d+)').astype('Int64')
    raw_data['assigned_to'] = raw_data['assigned_to'].str.extract('(\d+)').astype('Int64')
    raw_data['closed_code'] = raw_data['closed_code'].str.extract('(\d+)').astype('Int64')
    raw_data['resolved_by'] = raw_data['resolved_by'].str.extract('(\d+)').astype('Int64')

    # Convert "caller_id" column to numeric, replacing non-numeric values with NaN
    raw_data['caller_id'] = pd.to_numeric(raw_data['caller_id'].str.replace('Caller ', ''), errors='coerce')
    # Convert the coller_id to integeDataset Preview:rs
    raw_data['caller_id'] = raw_data['caller_id'].astype('Int64')
    # Extract numeric part from "number" and convert to integer
    raw_data['number'] = raw_data['number'].str.extract('(\d+)').astype('Int64')

    raw_data['resolution_time'] = (raw_data['resolved_at'] - raw_data['opened_at']).dt.total_seconds() / 3600  # in hours
    raw_data = raw_data[raw_data['resolved_at'] >= raw_data['opened_at']]

    # Remove rows with incident_state equal to '-100'
    raw_data = raw_data[raw_data['incident_state'] != '-100']

    # Drop rows with missing values
    raw_data = raw_data.dropna()
    raw_data.reset_index(drop=True, inplace=True)

    # Drop the columns
    raw_data = raw_data.drop(columns=columns_to_drop)

    # ------------- Machine Learning -------------------

    # Calculate mean for assigned_to and reassignment_count
    assigned_to_mean = raw_data['assigned_to'].mean()
    reassignment_count_mean = raw_data['reassignment_count'].mean()

    # Calculate mean for resolution_time
    resolution_time_mean = raw_data['resolution_time'].mean()

    # specified columns for fairness
    fairness_columns = ['assigned_to', 'reassignment_count', 'knowledge', 'u_priority_confirmation', 'resolution_time']

    # Define criteria for fairness using mean as thresholds
    fairness_thresholds = {
        'assigned_to_threshold': round(assigned_to_mean),  # Use rounded mean
        'reassignment_count_threshold': round(reassignment_count_mean),  # Use rounded mean
        'knowledge_threshold': False,  
        'u_priority_confirmation_threshold': False,  
        'resolution_time_threshold': resolution_time_mean  
    }

    # Create a binary target variable 'fairness' based on criteria
    raw_data['fairness'] = (
        (raw_data['assigned_to'] <= fairness_thresholds['assigned_to_threshold']) &
        (raw_data['reassignment_count'] <= fairness_thresholds['reassignment_count_threshold']) &
        (raw_data['knowledge'] >= fairness_thresholds['knowledge_threshold']) &
        (raw_data['u_priority_confirmation'] >= fairness_thresholds['u_priority_confirmation_threshold']) &
        (raw_data['resolution_time'] <= fairness_thresholds['resolution_time_threshold'])
    ).astype(int)

    cleaned_data = raw_data.copy()
    return cleaned_data


def plot_top_10_categories_pie_chart(df):
    st.markdown("<h5 style='background-color: #089bcc; padding: 10px; border-radius: 5px;color: white;'>Top 10 Categories Pie Chart</h5>",
                 unsafe_allow_html=True)
    top_categories = df['category'].value_counts().head(10)
    top_categories_df = pd.DataFrame({'Category': top_categories.index, 'Count': top_categories.values})
    # Concatenate 'Category' with category numbers
    top_categories_df['Category'] = 'Category ' + top_categories_df['Category'].astype(str)
    fig = px.pie(top_categories_df, 
                 values='Count', 
                 names='Category',
                 title='Top 10 Categories of Incidents',
                 )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

def plot_incident_state_distribution(df):
    st.markdown("<h5 style='background-color: #87CEEB; padding: 10px; border-radius: 5px;color: white;'>Incident State Distribution</h5>",
                unsafe_allow_html=True)
    incident_state_counts = df['incident_state'].value_counts().reset_index()
    incident_state_counts.columns = ['Incident State', 'Count']
    fig = px.bar(incident_state_counts, x='Incident State', y='Count',
                  title='Distribution of Incident States',
                  labels={'Incident State': 'Incident State', 'Count': 'Count'})
    st.plotly_chart(fig)


def plot_incident_priority_distribution(df):
    st.markdown("<h5 style='background-color: #D3D3D3; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Priority Levels</h5>",
                unsafe_allow_html=True)

    priority_counts = df['priority'].value_counts().reset_index()
    priority_counts.columns = ['Priority Level', 'Count']

    fig_priority = px.pie(priority_counts, values='Count', names='Priority Level',
                          title='Distribution of Incident Priority Levels',
                          labels={'Priority Level': 'Priority Level', 'Count': 'Count'})
    st.plotly_chart(fig_priority)

def plot_incident_impact_distribution(df):
    st.markdown("<h5 style='background-color: #FFB6C1; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Impact Levels</h5>",
                unsafe_allow_html=True)

    impact_counts = df['impact'].value_counts().reset_index()
    impact_counts.columns = ['Impact Level', 'Count']

    fig_impact = px.bar(impact_counts, x='Impact Level', y='Count',
                        title='Distribution of Incident Impact Levels',
                        labels={'Impact Level': 'Impact Level', 'Count': 'Count'})
    
    st.plotly_chart(fig_impact)

def plot_incident_urgency_distribution(df):
    st.markdown("<h5 style='background-color: #089bcc; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Urgency Levels</h5>",
                unsafe_allow_html=True)

    urgency_counts = df['urgency'].value_counts().reset_index()
    urgency_counts.columns = ['Urgency Level', 'Count']

    fig_urgency = px.bar(urgency_counts, x='Urgency Level', y='Count',
                         title='Distribution of Incident Urgency Levels',
                         labels={'Urgency Level': 'Urgency Level', 'Count': 'Count'},
                         color='Urgency Level')  # Specify color here
    st.plotly_chart(fig_urgency)


def plot_time_distribution_of_incidents(df):
    st.markdown("<h5 style='background-color: #f0ad4e; padding: 10px; border-radius: 5px;color: white;'>Time Distribution of Incidents</h5>",
                unsafe_allow_html=True)

    time_distribution = df.groupby(df['opened_at'].dt.date)['number'].count().reset_index()
    time_distribution.columns = ['Date', 'Incident Count']

    fig_time_distribution = px.line(time_distribution, x='Date', y='Incident Count',
                                    title='Time Distribution of Incidents',
                                    labels={'Date': 'Date', 'Incident Count': 'Incident Count'},
                                    color_discrete_sequence=['#17a2b8'])  # Specify color here
    st.plotly_chart(fig_time_distribution)


def plot_distribution_of_active_inactive_incidents(df):
    st.markdown("<h5 style='background-color: #5cb85c; padding: 10px; border-radius: 5px;color: white;'>Distribution of Active/Inactive Incidents</h5>",
                unsafe_allow_html=True)

    active_counts = df['active'].value_counts().reset_index()
    active_counts.columns = ['Active', 'Count']

    fig_active = px.pie(active_counts, values='Count', names='Active',
                        title='Distribution of Active/Inactive Incidents',
                        labels={'Active': 'Active', 'Count': 'Count'},
                        color='Active')  # Specify color here
    st.plotly_chart(fig_active)

def plot_incident_knowledge_distribution(df):
    st.markdown("<h5 style='background-color: #8A2BE2; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Knowledge</h5>",
                unsafe_allow_html=True)

    knowledge_counts = df['knowledge'].value_counts().reset_index()
    knowledge_counts.columns = ['Knowledge', 'Count']

    fig_knowledge = px.pie(knowledge_counts, values='Count', names='Knowledge',
                           title='Distribution of Incident Knowledge',
                           labels={'Knowledge': 'Knowledge', 'Count': 'Count'})
    st.plotly_chart(fig_knowledge)


def plot_assignment_group_distribution(df):
    st.markdown("<h5 style='background-color: #FF4500; padding: 10px; border-radius: 5px;color: white;'>Distribution of Assignment Groups</h5>",
                unsafe_allow_html=True)

    assignment_group_counts = df['assignment_group'].value_counts().reset_index()
    assignment_group_counts.columns = ['Assignment Group', 'Count']

    fig_assignment_group = px.pie(assignment_group_counts.head(20), values='Count', names='Assignment Group',
                                  title='Top 20 Assignment Groups')
    st.plotly_chart(fig_assignment_group)


def plot_opened_by_distribution(df):
    st.markdown("<h5 style='background-color: #1E90FF; padding: 10px; border-radius: 5px;color: white;'>Distribution of Opened By</h5>",
                unsafe_allow_html=True)

    opened_by_counts = df['opened_by'].value_counts().reset_index()
    opened_by_counts.columns = ['Opened By', 'Count']

    fig_opened_by = px.pie(opened_by_counts.head(20), values='Count', names='Opened By',
                           title='Top 20 Opened By')
    st.plotly_chart(fig_opened_by)


def plot_resolved_by_distribution(df):
    st.markdown("<h5 style='background-color: #228B22; padding: 10px; border-radius: 5px;color: white;'>Distribution of Resolved By</h5>",
                unsafe_allow_html=True)

    resolved_by_counts = df['resolved_by'].value_counts().reset_index()
    resolved_by_counts.columns = ['Resolved By', 'Count']

    fig_resolved_by = px.pie(resolved_by_counts.head(20), values='Count', names='Resolved By',
                             title='Top 20 Resolved By')
    st.plotly_chart(fig_resolved_by)


def plot_notification_preference_distribution(df):
    st.markdown("<h5 style='background-color: #FFD700; padding: 10px; border-radius: 5px;color: white;'>Notification Preference Distribution</h5>",
                unsafe_allow_html=True)

    notify_counts = df['notify'].value_counts().reset_index()
    notify_counts.columns = ['Notification Preference', 'Count']

    fig_notify = px.pie(notify_counts, names='Notification Preference', values='Count',
                        title='Notification Preference Distribution',
                        labels={'Notification Preference': 'Notification Preference', 'Count': 'Count'})
    st.plotly_chart(fig_notify)


def plot_incident_resolution_distribution(df):
    st.markdown("<h5 style='background-color: #FFA500; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Resolutions</h5>",
                unsafe_allow_html=True)

    resolved_counts = df['incident_state'].value_counts().reset_index()
    resolved_counts.columns = ['Incident Resolution', 'Count']

    fig_resolved = px.bar(resolved_counts, x='Incident Resolution', y='Count',
                          title='Distribution of Incident Resolutions',
                          labels={'Incident Resolution': 'Incident Resolution', 'Count': 'Count'})
    st.plotly_chart(fig_resolved)


def plot_incident_impact_distribution(df):
    st.markdown("<h5 style='background-color: #00CED1; padding: 10px; border-radius: 5px;color: white;'>Distribution of Impact Levels</h5>",
                unsafe_allow_html=True)

    impact_counts = df['impact'].value_counts().reset_index()
    impact_counts.columns = ['Impact Level', 'Count']

    fig_impact = px.bar(impact_counts, x='Impact Level', y='Count',
                        title='Distribution of Impact Levels',
                        labels={'Impact Level': 'Impact Level', 'Count': 'Count'})
    st.plotly_chart(fig_impact)


def plot_incident_reassignment_distribution(df):
    st.markdown("<h5 style='background-color: #800080; padding: 10px; border-radius: 5px;color: white;'>Distribution of Incident Reassignments</h5>",
                unsafe_allow_html=True)

    reassignment_counts = df['reassignment_count'].value_counts().nlargest(15).reset_index()
    reassignment_counts.columns = ['Reassignment Count', 'Count']

    fig_reassignment = px.bar(reassignment_counts, x='Reassignment Count', y='Count',
                              title='Distribution of Incident Reassignments',
                              labels={'Reassignment Count': 'Reassignment Count', 'Count': 'Count'})
    st.plotly_chart(fig_reassignment)


def plot_resolved_status_distribution(df):
    st.markdown("<h5 style='background-color: #4682B4; padding: 10px; border-radius: 5px; color: white;'>Distribution of Incident Resolutions</h5>",
                unsafe_allow_html=True)


    resolved_counts = df['incident_state'].apply(lambda x: 'Resolved' if x == 'Resolved' else 'Not Resolved').value_counts().reset_index()
    resolved_counts.columns = ['Resolution Status', 'Count']

    fig_resolved = px.pie(resolved_counts, names='Resolution Status', values='Count',
                          title='Distribution of Incident Resolutions',
                          labels={'Resolution Status': 'Resolution Status', 'Count': 'Count'})
    st.plotly_chart(fig_resolved)

def plot_incident_lifecycle_distribution(df):
    st.markdown("<h5 style='background-color: #4682B4; padding: 10px; border-radius: 5px; color: white;'>Distribution of Incident Lifecycle (Active/Closed)</h5>",
                unsafe_allow_html=True)

    active_closed_counts = df['active'].value_counts().reset_index()
    active_closed_counts.columns = ['Incident Lifecycle', 'Count']

    fig_lifecycle = px.pie(active_closed_counts, names='Incident Lifecycle', values='Count',
                           title='Distribution of Incident Lifecycle (Active/Closed)',
                           labels={'Incident Lifecycle': 'Incident Lifecycle', 'Count': 'Count'})
    st.plotly_chart(fig_lifecycle)


def plot_incident_categories_distribution(df):
    st.markdown("<h5 style='background-color: #FF6347; padding: 10px; border-radius: 5px; color: white;'>Distribution of Incident Categories</h5>",
                unsafe_allow_html=True)

    category_counts = df['category'].value_counts().reset_index().head(10)
    category_counts.columns = ['Category', 'Count']

    fig_category_distribution = px.pie(category_counts, values='Count', names='Category',
                                       title='Top 10 Incident Categories Distribution')
    st.plotly_chart(fig_category_distribution)



def plot_incident_states_time_series(df):
    st.markdown("<h5 style='background-color: #00FFFF; padding: 10px; border-radius: 5px; color: white;'>Time Series Plot of Incident States</h5>",
                unsafe_allow_html=True)

    state_counts = df['incident_state'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']

    fig_states = px.line(state_counts, x='State', y='Count',
                         title='Incident Lifecycle Analysis - Time Series Plot',
                         labels={'State': 'Incident State', 'Count': 'Count'})
    st.plotly_chart(fig_states)


def plot_priority_sunburst_chart(df):
    st.markdown("<h5 style='background-color: #800080; padding: 10px; border-radius: 5px; color: white;'>Priority Sunburst Chart</h5>",
                unsafe_allow_html=True)

    priority_sunburst = df.groupby(['category', 'priority']).size().reset_index(name='count')
    fig = px.sunburst(priority_sunburst, path=['category', 'priority'], values='count',
                      title='Priority Sunburst Chart')
    st.plotly_chart(fig)

def plot_incident_count_by_urgency_and_impact(df):
    fig = px.histogram(df, x='urgency', color='impact',
                       title='Incident Count by Urgency and Impact',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_sunburst_chart(df):
    fig = px.sunburst(df, path=['incident_state', 'priority', 'category'],
                      title='Incident State Distribution Sunburst Chart',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_priority_distribution_by_category(df):
    fig = px.histogram(df, x='category', color='priority',
                       title='Priority Distribution by Category',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_priority_distribution_by_impact(df):
    fig = px.histogram(df, x='impact', color='priority',
                       title='Priority Distribution by Impact',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_sunburst_chart_by_location(df):
    fig = px.sunburst(df, path=['incident_state', 'priority', 'location'],
                      title='Incident State Distribution Sunburst Chart by Location',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_incident_state_distribution_by_category(df):
    fig = px.histogram(df, x='incident_state', color='category',
                       title='Incident State Distribution by Category',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_treemap_by_subcategory(df):
    fig = px.treemap(df, path=['incident_state', 'subcategory'],
                     title='Incident State Distribution Treemap by Subcategory',
                     height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_by_assignment_group_and_impact(df):
    fig = px.sunburst(df, path=['assignment_group', 'impact', 'incident_state'],
                      title='Incident State Distribution by Assignment Group and Impact',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_incident_priority_sunburst_chart_by_assignment_group_and_location(df):
    fig = px.sunburst(df, path=['assignment_group', 'location', 'priority'],
                      title='Incident Priority Sunburst Chart by Assignment Group and Location',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)

def plot_reassignment_count_distribution_by_priority(df):
    fig = px.histogram(df, x='reassignment_count', color='priority',
                       title='Distribution of Reassignment Count by Priority',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_by_knowledge_and_category(df):
    fig = px.sunburst(df, path=['knowledge', 'category', 'incident_state'],
                      title='Incident State Distribution by Knowledge and Category',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_incident_state_distribution_by_urgency_and_priority(df):
    fig = px.sunburst(df, path=['urgency', 'priority', 'incident_state'],
                      title='Incident State Distribution by Urgency and Priority',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_priority_distribution_by_contact_type(df):
    fig = px.histogram(df, x='contact_type', color='priority',
                       title='Incident Priority Distribution by Contact Type',
                       labels={'count': 'Incident Count'},
                       height=600, width=800)
    st.plotly_chart(fig)


def plot_incident_state_distribution_by_impact_and_urgency(df):
    fig = px.sunburst(df, path=['impact', 'urgency', 'incident_state'],
                      title='Incident State Distribution by Impact and Urgency',
                      labels={'value': 'Incident Count'},
                      height=600)
    st.plotly_chart(fig)


def plot_distribution_of_incidents_across_assignment_groups(df):
    assignment_group_counts = df['assignment_group'].value_counts()
    fig = px.bar(assignment_group_counts, x=assignment_group_counts.index, y=assignment_group_counts.values,
                 labels={'x': 'Assignment Group', 'y': 'Count'},
                 title='Distribution of Incidents Across Assignment Groups')
    fig.update_layout(xaxis_title='Assignment Group', yaxis_title='Count', xaxis_tickangle=-45)
    st.plotly_chart(fig)


def plot_distribution_of_incident_impact_levels(df):
    impact_counts = df['impact'].value_counts()
    fig = px.pie(impact_counts, names=impact_counts.index, values=impact_counts.values,
                 title='Distribution of Incident Impact Levels',
                 hole=0.4)
    st.plotly_chart(fig)


def plot_distribution_of_incident_urgency_levels(df):
    urgency_counts = df['urgency'].value_counts()
    fig = px.pie(urgency_counts, names=urgency_counts.index, values=urgency_counts.values,
                 title='Distribution of Incident Urgency Levels',
                 hole=0.4)
    st.plotly_chart(fig)


def plot_top_20_location_distribution_of_incidents(df):
    location_counts = df['location'].value_counts()
    fig = px.pie(location_counts.head(20), values=location_counts.head(20).values,
                 names=location_counts.head(20).index,
                 title='Top 20 Locations - Distribution of Incidents')
    st.plotly_chart(fig)


def generate_fairness_plot(df):
    # Group by 'assigned_to' and calculate average resolution time
    assigned_to_analysis = df.groupby('assigned_to')['resolution_time'].mean().reset_index()

    # Create a fairness plot
    fig = px.bar(assigned_to_analysis, x='assigned_to', y='resolution_time',
                 labels={'assigned_to': 'Assigned To', 'resolution_time': 'Average Resolution Time'},
                 title='Fairness Analysis for Assigned To',
                 color='assigned_to', color_continuous_scale='viridis')
    return fig

def generate_category_fairness_plot(df):
    # Group by 'category' and calculate average resolution time
    category_analysis = df.groupby('category')['resolution_time'].mean().reset_index()

    # Create a fairness plot
    fig = px.bar(category_analysis, x='category', y='resolution_time',
                 labels={'category': 'Category', 'resolution_time': 'Average Resolution Time'},
                 title='Fairness Analysis for Category',
                 color='category', color_continuous_scale='viridis')
    return fig



def plot_assignment_group_distribution(df):
    # Create a DataFrame for the assignment group distribution
    assignment_group_distribution = df['assignment_group'].value_counts(normalize=True)
    df_assignment_group_distribution = assignment_group_distribution.reset_index()
    df_assignment_group_distribution.columns = ['Assignment Group', 'Proportion']

    # Plot the distribution using Plotly
    fig = px.bar(df_assignment_group_distribution, x='Assignment Group', y='Proportion',
                 title='Distribution of Incidents Across Assignment Groups',
                 labels={'Proportion': 'Proportion of Incidents', 'Assignment Group': 'Assignment Group'},
                 template='plotly_dark')

    # Customize layout
    fig.update_layout(xaxis=dict(type='category'), xaxis_title_text='Assignment Group')
    return fig




# Pages


def page_1():
    # Apply CSS styling for the header background
    st.markdown(
        """
        <style>
            .header-background {
                background-color: #008080; /* Background color */
                color: white; /* Text color */
                padding: 10px; /* Padding */
                border-radius: 10px; /* Rounded corners */
                margin-bottom: 20px; /* Bottom margin */
                text-align: center; /* Center-align text */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header with background
    st.markdown("<h1 class='header-background'>Dataset Information</h1>", unsafe_allow_html=True)
    
    # Read attribute information from the text file
    with open("Incident_response.txt", "r") as file:
        attribute_info = file.read()

    # Display the attribute information in a styled box
    st.markdown(
        f"""
        <div class="attribute-info">
            <h2>Attribute Information:</h2>
            <pre>{attribute_info}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Read the dataset
    df = clean_dataset()
    st.markdown("---")
    # Display the dataset if it's successfully loaded
    if df is not None:

        st.markdown("<h1 style='text-align: center; color: white; background-color: #008080;'>Cleaned Dataset Preview:</h1>",
                unsafe_allow_html=True)
        st.write(df.head())

    # Apply CSS styling
    st.markdown(
        """
        <style>
            .attribute-info {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                overflow: auto;
            }
            .attribute-info h2 {
                color: #008080;
            }
            .attribute-info pre {
                font-size: 16px;
                white-space: pre-wrap;
            }
        </style>
        """,
        unsafe_allow_html=True
    )






def page_2(df):
    st.header("Exploratory Data Analysis")

    # Dropdown menu in the sidebar
    st.sidebar.title("EDA Options")

    eda_options = ["Top 10 Categories Pie Chart", 
                "Incident State Distribution", 
                "Distribution of Incident Priority Levels",
                "Distribution of Incident Impact Levels",
                "Time Distribution of Incidents",
                "Distribution of Active/Inactive Incidents",
                "Distribution of Incident Knowledge",
                "Distribution of Assignment Groups",
                "Distribution of Opened By",
                "Distribution of Resolved By",
                "Notification Preference Distribution",
                "Distribution of Incident Resolutions",
                "Time Distribution of Incidents",
                "Distribution of Incident Reassignments",
                "Distribution of Resolved Status",
                "Distribution of Incident Lifecycle (Active/Closed)",
                "Distribution of Incident Categories",
                "Priority Sunburst Chart",
                "Time Series Plot of Incident States",
                "Incident Count by Urgency and Impact",
                "Incident State Distribution Sunburst Chart",
                "Priority Distribution by Category",
                "Priority Distribution by Impact",
                "Incident State Distribution Sunburst Chart by Location",
                "Incident State Distribution by Category",
                "Incident State Distribution Treemap by Subcategory",
                "Incident State Distribution by Assignment Group and Impact",
                "Incident Priority Sunburst Chart by Assignment Group and Location",
                "Distribution of Reassignment Count by Priority",
                "Incident State Distribution by Knowledge and Category",
                "Incident State Distribution by Urgency and Priority",
                "Incident Priority Distribution by Contact Type",
                "Incident State Distribution by Impact and Urgency",
                "Distribution of Incidents Across Assignment Groups",
                "Distribution of Incident Impact Levels",
                "Distribution of Incident Urgency Levels",
                "Top 20 Locations - Distribution of Incidents"]



    selected_option = st.sidebar.selectbox("Select EDA Option", eda_options)
    # st.write(f"Selected EDA Option: {selected_option}")

    show_all_plots = st.sidebar.checkbox("Show All Plots", False)

    if show_all_plots:
        plot_top_10_categories_pie_chart(df)
        plot_incident_state_distribution(df)
        plot_incident_priority_distribution(df)
        plot_incident_impact_distribution(df)
        plot_time_distribution_of_incidents(df)
        plot_distribution_of_active_inactive_incidents(df)
        plot_incident_knowledge_distribution(df)
        plot_assignment_group_distribution(df)
        plot_opened_by_distribution(df)
        plot_resolved_by_distribution(df)
        plot_notification_preference_distribution(df)
        plot_incident_resolution_distribution(df)
        plot_incident_reassignment_distribution(df)
        plot_resolved_status_distribution(df)
        plot_incident_lifecycle_distribution(df)
        plot_incident_categories_distribution(df)
        plot_priority_sunburst_chart(df)
        plot_incident_states_time_series(df)
        plot_incident_count_by_urgency_and_impact(df)
        plot_incident_state_distribution_sunburst_chart(df)
        plot_priority_distribution_by_category(df)
        plot_priority_distribution_by_impact(df)
        plot_incident_state_distribution_sunburst_chart_by_location(df)
        plot_incident_state_distribution_by_category(df)
        plot_incident_state_distribution_treemap_by_subcategory(df)
        plot_incident_state_distribution_by_assignment_group_and_impact(df)
        plot_incident_priority_sunburst_chart_by_assignment_group_and_location(df)
        plot_reassignment_count_distribution_by_priority(df)
        plot_incident_state_distribution_by_knowledge_and_category(df)
        plot_incident_state_distribution_by_urgency_and_priority(df)
        plot_priority_distribution_by_contact_type(df)
        plot_incident_state_distribution_by_impact_and_urgency(df)
        plot_distribution_of_incidents_across_assignment_groups(df)
        plot_distribution_of_incident_impact_levels(df)
        plot_distribution_of_incident_urgency_levels(df)
        plot_top_20_location_distribution_of_incidents(df)
    else:
        match selected_option:
            case "Top 10 Categories Pie Chart":
                plot_top_10_categories_pie_chart(df)

            case "Incident State Distribution":
                plot_incident_state_distribution(df)

            case "Distribution of Incident Priority Levels":
                plot_incident_priority_distribution(df)

            case "Distribution of Incident Impact Levels":
                plot_incident_impact_distribution(df)

            case "Time Distribution of Incidents":
                plot_time_distribution_of_incidents(df)

            case "Distribution of Active/Inactive Incidents":
                plot_distribution_of_active_inactive_incidents(df)

            case "Distribution of Incident Knowledge":
                plot_incident_knowledge_distribution(df)

            case "Distribution of Assignment Groups":
                plot_assignment_group_distribution(df)

            case "Distribution of Opened By":
                plot_opened_by_distribution(df)

            case "Distribution of Resolved By":
                plot_resolved_by_distribution(df)

            case "Notification Preference Distribution":
                plot_notification_preference_distribution(df)

            case "Distribution of Incident Resolutions":
                plot_incident_resolution_distribution(df)

            case "Distribution of Incident Reassignments":
                plot_incident_reassignment_distribution(df)

            case "Distribution of Resolved Status":
                plot_resolved_status_distribution(df)

            case "Distribution of Incident Lifecycle (Active/Closed)":
                plot_incident_lifecycle_distribution(df)

            case "Distribution of Incident Categories":
                plot_incident_categories_distribution(df)

            case "Priority Sunburst Chart":
                plot_priority_sunburst_chart(df)

            case "Time Series Plot of Incident States":
                plot_incident_states_time_series(df)

            case "Incident Count by Urgency and Impact":
                plot_incident_count_by_urgency_and_impact(df)

            case "Incident State Distribution Sunburst Chart":
                plot_incident_state_distribution_sunburst_chart(df)

            case "Priority Distribution by Category":
                plot_priority_distribution_by_category(df)

            case "Priority Distribution by Impact":
                plot_priority_distribution_by_impact(df)

            case "Incident State Distribution Sunburst Chart by Location":
                plot_incident_state_distribution_sunburst_chart_by_location(df)

            case "Incident State Distribution by Category":
                plot_incident_state_distribution_by_category(df)

            case "Incident State Distribution Treemap by Subcategory":
                plot_incident_state_distribution_treemap_by_subcategory(df)

            case "Incident State Distribution by Assignment Group and Impact":
                plot_incident_state_distribution_by_assignment_group_and_impact(df)

            case "Incident Priority Sunburst Chart by Assignment Group and Location":
                plot_incident_priority_sunburst_chart_by_assignment_group_and_location(df)

            case "Distribution of Reassignment Count by Priority":
                plot_reassignment_count_distribution_by_priority(df)

            case "Incident State Distribution by Knowledge and Category":
                plot_incident_state_distribution_by_knowledge_and_category(df)

            case "Incident State Distribution by Urgency and Priority":
                plot_incident_state_distribution_by_urgency_and_priority(df)

            case "Incident Priority Distribution by Contact Type":
                plot_priority_distribution_by_contact_type(df)

            case "Incident State Distribution by Impact and Urgency":
                plot_incident_state_distribution_by_impact_and_urgency(df)

            case "Distribution of Incidents Across Assignment Groups":
                plot_distribution_of_incidents_across_assignment_groups(df)

            case "Distribution of Incident Impact Levels":
                plot_distribution_of_incident_impact_levels(df)

            case "Distribution of Incident Urgency Levels":
                plot_distribution_of_incident_urgency_levels(df)

            case "Top 20 Locations - Distribution of Incidents":
                plot_top_20_location_distribution_of_incidents(df)

            case _:
                st.write("No chart selected.")



def page_3(df):
    st.markdown("<h1 style='text-align: center; color: white; background-color: #008080;'>Fairness Analysis</h1>",
                 unsafe_allow_html=True)

    st.markdown("""
    <style>
        .description {
            font-size: 18px;
            color: #333;
            margin-bottom: 20px;
        }
        .columns {
            font-size: 16px;
            color: #333;
            margin-bottom: 10px;
            font-weight: bold;
            padding: 5px;
            border-radius: 5px;
            display: inline-block;
        }
        .column-1 { background-color: #E6F2FF; }
        .column-2 { background-color: #CCEDFF; }
        .column-3 { background-color: #B3E0FF; }
        .column-4 { background-color: #99D4FF; }
        .column-5 { background-color: #80C9FF; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<p class='description'>This analysis evaluates the fairness of the assignment process by analyzing various factors.</p>", unsafe_allow_html=True)

    # Display column names with different background colors
    st.markdown("<p class='columns column-1'>Columns used for analysis:</p>", unsafe_allow_html=True)
    st.markdown("<ul>", unsafe_allow_html=True)
    fairness_columns = ['assigned_to', 'reassignment_count', 'knowledge', 'u_priority_confirmation', 'resolution_time']
    for idx, column in enumerate(fairness_columns, start=1):
        st.markdown(f"<li class='columns column-{idx}'>{column}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 2px solid #ccc;'>", unsafe_allow_html=True)

    # Create a subset DataFrame for fairness analysis
    fairness_df = df[fairness_columns].copy()

    for idx, column in enumerate(fairness_columns[1:], start=1):
        # Create contingency table for statistical test
        contingency_table = pd.crosstab(fairness_df['assigned_to'], fairness_df[column])

        # Perform chi-square test for independence
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

        # Calculate fairness metric (Chi-square value)
        fairness_metric = chi2_stat / df.shape[0]

        # Format the fairness assessment with colored keywords, boldness, size, and background color
        fairness_assessment = (
            f"<div style='background-color: {'#f0f0f0' if idx % 2 == 0 else '#e6f2ff'}; padding: 10px;'>"
            f"<span style='color: #007acc; font-weight: bold; font-size: 20px;'>Fairness Assessment</span> "
            f"between <span style='color: #007acc; font-weight: bold; font-size: 20px;'>assigned_to</span> "
            f"and <span style='color: #007acc; font-weight: bold; font-size: 20px;'>{column}</span>:<br>"
            f"<span style='color: #007acc;'>Chi-Square Statistic</span>: {chi2_stat}<br>"
            f"<span style='color: #007acc;'>P-value</span>: {p_value}<br>"
            f"<span style='color: #007acc;'>Fairness Metric</span>: {fairness_metric}<br>"
        )

        # Check for fairness based on a significance level (0.05)
        if p_value < 0.05:
            fairness_assessment += (
                f"<span style='color: #cc0000; font-weight: bold;'>"
                f"There is evidence of potential unfairness</span> between "
                f"<span style='color: #007acc; font-weight: bold;'>assigned_to</span> and "
                f"<span style='color: #007acc; font-weight: bold;'>{column}</span>.<br>"
            )
        else:
            fairness_assessment += (
                f"<span style='color: #009900; font-weight: bold;'>"
                f"There is no evidence of unfairness</span> between "
                f"<span style='color: #007acc; font-weight: bold;'>assigned_to</span> and "
                f"<span style='color: #007acc; font-weight: bold;'>{column}</span>.<br>"
            )

        # Close the <div> tag for this section
        fairness_assessment += "</div>"

        # Write fairness assessment with HTML formatting and horizontal line
        st.markdown(fairness_assessment, unsafe_allow_html=True)
        st.markdown("<hr style='border-top: 2px solid #ccc;'>", unsafe_allow_html=True)

    fig = generate_fairness_plot(df)
    fig_2 = generate_category_fairness_plot(df)
    fig_3 = plot_assignment_group_distribution(df)
    st.plotly_chart(fig)
    st.plotly_chart(fig_2)
    st.plotly_chart(fig_3)



# Define a function to load data
@st.cache_resource(show_spinner = True)
def load_data():
    df = clean_dataset()  # Load the dataset
    return df

# Define a function to preprocess the data
@st.cache_resource(show_spinner = True)
def preprocess_data(df):
    # Select relevant features and target variable
    X = df[['reassignment_count', 'knowledge', 'u_priority_confirmation', 'resolution_time',
            'impact', 'urgency', 'priority']]
    y = df['fairness']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a ColumnTransformer to handle one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('priority', OneHotEncoder(), ['priority']),
            ('impact', OneHotEncoder(), ['impact']),
            ('urgency', OneHotEncoder(), ['urgency']),
        ],
        remainder='passthrough'
    )

    # Create a RandomForestClassifier pipeline
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    clf.fit(X_train, y_train)

    return clf, X_test, y_test

# Define a function to evaluate new data
@st.cache_resource(show_spinner = True)
def evaluate_new_data(_clf, new_data):
    # Convert new data to DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Make prediction using trained model
    prediction = _clf.predict(new_data_df)

    # Map prediction result to "Fair" or "Unfair"
    prediction_label = "Fair" if prediction[0] == 1 else "Unfair"

    return prediction_label

# Define the page_4 function
def page_4(df):
    st.write("<h1 style='text-align: center; color: #008080;'>Machine Learning</h1>", unsafe_allow_html=True)
    st.write("<div class='content'>", unsafe_allow_html=True)

    # Description about the target and features
    st.write("<div class='description'>", unsafe_allow_html=True)
    st.write("<h2>Description:</h2>", unsafe_allow_html=True)
    st.write("<p>The target variable in this analysis is 'fairness', representing the fairness of the assignment process. The features used for prediction include 'reassignment_count', 'knowledge', 'u_priority_confirmation', 'resolution_time', 'impact', 'urgency', and 'priority'.</p>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)

    clf, X_test, y_test = preprocess_data(df)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy with style
    st.write("<div class='result'>", unsafe_allow_html=True)
    st.write("<h2>Random Forest Model Performance:</h2>", unsafe_allow_html=True)
    st.write(f"<p style='font-size: 24px; color: #333;'>Accuracy: <span style='color: #008080;'>{accuracy * 100:.2f}%</span></p>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)

    # Input section for new data
    st.write("<div class='input-section'>", unsafe_allow_html=True)
    st.write("<h2>Evaluate New Data:</h2>", unsafe_allow_html=True)

    # Default values for new data
    default_values = {
        'reassignment_count': 1,
        'knowledge': True,
        'u_priority_confirmation': True,
        'resolution_time': 10.0,
        'impact': '2 - Medium',
        'urgency': '2 - Medium',
        'priority': '3 - Moderate'
    }

    # Input form for new data
    new_data = {}
    for column, default_value in default_values.items():
        if isinstance(default_value, bool):
            new_data[column] = st.checkbox(column, value=default_value, key=column)
        else:
            if column in ['impact', 'urgency', 'priority']:
                options = ['2 - Medium', '3 - Low', '1 - High'] if column != 'priority' else ['3 - Moderate', 
                                                                                              '4 - Low', 
                                                                                              '2 - High',
                                                                                              '1 - Critical']
                new_data[column] = st.selectbox(column, options, index=options.index(default_value))
            else:
                new_data[column] = st.text_input(column, value=default_value, key=column)

    # Evaluate fairness of new data
    if st.button("Evaluate Fairness"):
        prediction_label = evaluate_new_data(clf, new_data)

        # Display prediction result
        st.write("<div class='prediction'>", unsafe_allow_html=True)
        st.write("<h3>Prediction:</h3>", unsafe_allow_html=True)
        st.write(f"<p style='font-size: 18px; color: #333;'>The predicted fairness for the new data is <span style='color: #008080;'>{prediction_label}</span>.</p>", unsafe_allow_html=True)
        st.write("</div>", unsafe_allow_html=True)

    st.write("</div>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)



def main():
    # st.title("Streamlit Application")


    # Apply CSS styling for the title background
    st.markdown(
        """
        <style>
            .title-background {
                background-color: #9042f5; /* Background color */
                color: white; /* Text color */
                padding: 20px; /* Padding */
                border-radius: 10px; /* Rounded corners */
                text-align: center; /* Center-align text */
                font-family: 'Arial Black', sans-serif; /* Font family */
                font-size: 36px; /* Font size */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title with background and styled font
    st.markdown("<h1 class='title-background'>Incident Response Fairness Analysis</h1>", unsafe_allow_html=True)


    st.markdown("<hr style='height: 4px; background-color: #ccc;'>", unsafe_allow_html=True)

    st.sidebar.title("Choose a page to view more details.")

    page_options = ["Data Preparation", "Exploratory Data Analysis", "Fairness Analysis", "Machine Learning"]
    selected_page = st.sidebar.radio("Select Page", page_options)

    df = clean_dataset()  # Load the dataset

    # Main page content
    if selected_page == "Data Preparation":
        page_1()
    elif selected_page == "Exploratory Data Analysis":
        page_2(df)
    elif selected_page == "Fairness Analysis":
        page_3(df)
    elif selected_page == "Machine Learning":
        page_4(df)

    # Information
    st.sidebar.markdown("---")
    st.sidebar.title("Developers")
    st.sidebar.markdown("- **Mohsen Maaleki**")

    st.sidebar.title("Links")
    github_link = "[GitHub](https://github.com/your_username)"
    dataset_link = "[Kaggle Dataset](https://www.kaggle.com/datasets/vipulshinde/incident-response-log)"
    st.sidebar.markdown(github_link, unsafe_allow_html=True)
    st.sidebar.markdown(dataset_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()