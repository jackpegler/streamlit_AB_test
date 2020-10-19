import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ttest_ind
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image


st.title("Significance Test")
st.markdown("#### An app to help test significnace of A/B Tests (2 groups)")
st.markdown("""* Testing two proportions (e.g. % Engaged in a Promotion)
* Testing 2 means (e.g. $ amount paid)""")

st.markdown("## **1. Load your data**")
st.markdown("""
The data should be in the following format:
* A `.csv` file
* A column with **group labels** (e.g. Treatment, Control)
* *[For Proportion Test]* A column to show **if engaged or not** (e.g. 1=Engaged, 0=Did Not Engage)
* *[For Difference of Means Test]* At least one column (can have several) with the **value of numeric metric** (e.g. amount paid) to test
* Try not to have `NULL` values
""")

### FIRST GET THE DATA
# DISABLE WARNING
st.set_option('deprecation.showfileUploaderEncoding', False)
### PROMPT USER TO UPLOAD THEIR FILE
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

### SHOW DATA THE USER LOADS
if uploaded_file is not None:
    @st.cache
    def load_data():
        data = pd.read_csv(uploaded_file)
        return data

    data = load_data()

    show_data = st.checkbox("Show RAW data of {} records".format(len(data)), True)
    if show_data:
        st.subheader("Your raw data")
        st.write(data)


st.markdown("## **2. Pick your test**")

### OPTIN 1: DIFFERENCE OF TWO PROPORTIONS
proportion_test = st.checkbox("OPTION 1: Test two proportions", False)

if proportion_test:
    ### Pick the columns for the test
    column_options_prop = data.columns

    st.write("Select relevant columns from raw data")
    groups = st.selectbox("Select column with group labels", column_options_prop, key = "col_for_groups_prop")
    engaged = st.selectbox("Select column with engaged or not", column_options_prop, key="col_for_engaged_prop")

    st.sidebar.markdown("### **Configue Chi-Square for difference of proportions:**")
    alpha_prop = st.sidebar.slider("Alpha:", 0.0, 1.0, 0.05, key="alpha_prop")
    critical_val = st.sidebar.slider("Critcal Value:", 0.0, 11.34, 3.84, key="crtical_val_prop")


    prop_eval_metric = st.radio("choose how to evaluate:",['Alpha', 'Critical Value'], key="prop_eval_metric")

    if st.button("Test Significance",key='proportion_test'):

        st.markdown("We test the significance using a **Chi-square test**")

        ### EXPLAINING CHI SQUARE TEST
        st.write("The default set up is for a 2dof and 5%. The defaults can be changed in the sidebar, to learn more expand the explaination section AT THE BOTTOM")

        ### PLOT SUMMARY stats
        st.subheader("Summary Stats")
        engagement_summary = data[[engaged,groups]].groupby(by=groups).agg(['mean','std','count'])
        st.write(engagement_summary)


        ### Plot totals
        st.subheader("Totals Engaged")
        actuals = data.pivot_table(index=groups,columns=engaged, aggfunc='size')
        st.write(actuals)

        ### PLOT ENGAGEMENT RATE

        ### FIRST GET ENAGEMENT RATE (ONLY WORKS FOR 2 GROUPS)
        group_names = data[groups].unique()
        group_engagement = [data[data[groups]==group_names[i]][engaged].mean() for i in range(0,len(group_names))]

        plot_data = pd.DataFrame.from_dict({'groups':group_names,'engagement':group_engagement})
        ### Allow Sample of the numbers
        st.text("Select number of values to plot")
        sample_length = st.slider("Values to sample:", 0, len(plot_data), len(plot_data))
        plot_data_sample = plot_date.sample(n=sample_length, random_state=42)
        ### PLOT IT
        st.subheader("Engagement Rate")
        fig1 = px.bar(plot_data_sample, x='groups', y='engagement', color='groups')
        st.plotly_chart(fig1, use_container_width=True)


        ### Test Significance with Chi Squared
        st.subheader("Test Significance with Chi Squared")

        def calculate_expected_2d(actuals):
            """
            Takes in the actuals number of customers in each group and computes the expected values.
            Works for simple set up of one experiment variable for treatment & control groups

            Input:
                - actuals: pd.dataframe of the actual results form the test

            Output:
                - expected: outputs pd.dataframe of the EXPECTED results in each category
            """

            expected = actuals.copy()

            grand_total = actuals.sum().sum()
            c_1 = actuals.iloc[:,0].sum()
            c_2 = actuals.iloc[:,1].sum()
            r_1 = actuals.iloc[0,:].sum()
            r_2 = actuals.iloc[1,:].sum()

            expected.iloc[0,0] = (r_1 * c_1) / grand_total
            expected.iloc[0,1] = (r_1 * c_2) / grand_total
            expected.iloc[1,0] = (r_2 * c_1) / grand_total
            expected.iloc[1,1] = (r_2 * c_2) / grand_total

            return expected



        expected = calculate_expected_2d(actuals)
        test_stat, pval = chisquare(f_obs=actuals, f_exp=expected, ddof=1, axis=None)

        test_stat = round(test_stat,0)
        pval = round(pval,3)

        st.write("The test statistic is: {} \n".format(test_stat))
        st.write("The P-Values is: {} \n".format(pval))


        if prop_eval_metric == 'Critical Value':
            if test_stat >= critical_val:
                st.markdown("**Our test stat of {0} >= {1} (Critical Value)** \n".format(test_stat, critical_val))
                st.success("Therefore the results are SIGNIFICANT and we REJECT H_0 and ACCEPT H_a")
                st.balloons()
            else:
                st.write("**Our test stat of {0} < {1} (Critical Value)** \n".format(test_stat, critical_val))
                st.warning("Therefore the results are NOT SIGNIFICANT and we KEEP H_0")

        elif prop_eval_metric == 'Alpha':
            if pval <= alpha_prop:
                st.markdown("**Our P-Value of {0} <= {1} (Alpha)** \n".format(pval, alpha_prop))
                st.success("Therefore the results are SIGNIFICANT and we REJECT H_0 and ACCEPT H_a")
                st.balloons()
            else:
                st.markdown("**Our P-Value of {0} > {1} (Alpha)** \n".format(pval, alpha_prop))
                st.warning("Therefore the results are NOT SIGNIFICANT and we KEEP H_0")



### OPTION 2: DIFFERENCE OF TWO MEANS
means_test = st.checkbox("OPTION 2: Test two means", False)

if means_test:

    st.markdown("## Testing two mean")

    ### PICK COLUMN DATA
    column_options_mean = data.columns
    groups = st.selectbox("Select column with group labels", column_options_mean, key = "col_for_groups_mean")
    metric = st.selectbox("Select column with numerical metric to test", column_options_mean, key="col_metric_mean")

    ### LET USER CHANGE ALPHA VALUE IF THEY WANT
    st.sidebar.markdown("### **Configue T-Test for difference of means:**")
    alpha_mean=st.sidebar.slider("Alpha:", 0.0, 1.0, 0.05, key="alpha_mean")


    ### GET GROUP NAMES AND SPLIT THE DATA
    group_names = list(set(data[groups]))
    Sample_A = data[data[groups]==group_names[0]][metric]
    Sample_B = data[data[groups]==group_names[1]][metric]


    if st.checkbox("Select sample data to plot (good if have large data set!)", False):
        ### Allow Sample of the numbers
        st.markdown("Select number of values to plot for each group")
        sample_1_length = st.slider("{}:".format(group_names[0]), 0, len(data[data[groups]==group_names[0]]), len(data[data[groups]==group_names[0]]))
        sample_2_length = st.slider("{}:".format(group_names[1]), 0, len(data[data[groups]==group_names[1]]), len(data[data[groups]==group_names[1]]))



    if st.button("Test Significance",key='mean_test'):
        st.markdown("We test the significance using a **T-Test**")
        st.markdown("The standard value of alpha is {}. The defaults can be changed in the sidebar, to learn more expand the explaination section at the bottom".format(alpha_mean))


        ### PLOT SUMMARY STATS
        st.markdown("Testing the significance of the change in **{}**".format(metric))
        metric_summary = data[[metric,groups]].groupby(by=groups).agg(['mean','std','count'])
        st.subheader("Summary Stats")
        st.write(metric_summary)

        ### PLOT SPREAD OF DATA
        st.subheader("Distribution of the groups")
        with st.spinner("Creating plot of data... (can take a minute if lots of data)"):
            g1 = data[data[groups]==group_names[0]][metric].sample(n=sample_1_length, random_state=42)
            g2 = data[data[groups]==group_names[1]][metric].sample(n=sample_2_length, random_state=42)

            hist_data = [g1,g2]
            fig = ff.create_distplot(hist_data, group_names)
            st.plotly_chart(fig, use_container_width=True)

            ### DO THE T-TEST
            t_result = ttest_ind(Sample_A, Sample_B)

            ### DISPLAY RESULTS OF THE TEST
            if (t_result[1] < alpha_mean):
                st.text("test statistic: {0}, p-value: {1}".format(t_result[0].round(3),t_result[1].round(3)))
                st.text("Alpha is {} so".format(alpha_mean))
                st.success("Difference is SIGNIFICANT")
                st.balloons()
            else:
                st.text("test statistic: {0}, p-value: {1}".format(t_result[0].round(3),t_result[1].round(3)))
                st.warning("Difference is NOT SIGNIFICANT")

st.markdown("## **3. More info on the tests**")
### EXPLAINER ON THE CHI SQUARE
if st.checkbox("Learn more about Chi-Square Test"):

    st.markdown("""
    ### Chi Square:

    #### _Hypothesis_
    * **H_0 (Null Hypothesis)** - There is no impact of the Treatment on the number of people engaging
    * **H_a (Alternate Hypothesis** - H_0 is FALSE and the Treatment does make a difference
    * Lets use _alpha = 0.05_ for our **significance**

    #### _Test Statistic_

    As we're looking at two categorical variables (engagment: yes/no & group: treatment/control) we use the Chi Squared test, which uses the following formula

    """)

    image1 = Image.open('images/equation_image7.jpg')
    st.image(image1, caption='Equation for Chi Square', use_column_width=False)


    st.markdown("""
    #### _Decision Rule_

    Our **degrees of freedom** are `(#row - 1) * (#column - 1) = (2-1) * (2-1) = 1`

    At our significance level of `alpha = 0.05` the critical value is **3.84**
    """)

    image2 = Image.open('images/chi-square_table.png')
    st.image(image2, caption='Chi Square distribution table', use_column_width=True)

    st.markdown("""
    Therefore we'll reject H_0 if our `Chi**2 >= 3.84`
    (Equivalently, if our returned p-value < alpha then we'll also reject the H_0)
    """)



    st.markdown("""
    #### _Computing the Test Statistic_

    To compute using the fomula above we need the expected values for each cell in the table, calculated using
    """)

    image3 = Image.open('images/equation.png')
    st.image(image3, caption='Formula for expected values', use_column_width=False)






## EXPLAINER ON THE T-TEST
if st.checkbox("Learn more about T-Test"):
    st.write("explain it alll")




## TODO: let you filter based on another field e.g. in certain days locked and also add visualisation of grpah ttest
## TODO: add optional explination of critical value to choose aswell
