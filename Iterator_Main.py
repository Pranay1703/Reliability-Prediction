import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import datetime
import logging
import warnings
import lifelines
import math
from sklearn.svm import SVR
import SVM_Model
#import tensorflow as tf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def iterator(unq_model_wise_complaints, raw_data_prepro, que, batch_selection, hzd_ckh, rgt_cens, BOBJ_CRM_max_kms):

    results_final_temp = pd.DataFrame \
        (columns=['Model' ,'Sub-Model', 'Dummy complaint code' ,'Selected_Month' ,'Usage_per_Month' ,'Shape', 'Scale', 'p_Value', 'N_DataPoints', 'Regression_Intercept' ,'Regression Score(%)_Training'
                 ,'Regression Score(%)_Testing' \ ,'LR_3_MIS_IPTV','LR_12_M IS_IPTV','TF_DL_3 _MIS','TF_DL_1 2_MIS','TF_DL_T raining_Score','TF_DL_T
                 esting_Score','SVM_3_M IS','SVM_12_ MIS','SVM_Tra ining_Score','SVM_Tes ting_Score' \
        ,
            'N_DataPoints_RGT_Cens', 'Regression Score(%)_Train_RGT_KMF', 'Regression Score(%)_Test_RGT_KM
                 ', 'Shape_RGT_KMF', 'Scale_RGT_KMF', 'p_Value_RGT_KMF', 'Regression_Intercept_RGT_KM
                 F','3 MIS IPTV_RGT_KMF', '12 MIS IPTV_RGT_KMF' \
        ,'Regression Score(%)_RGT_Benards', 'Shape_RGT_Benards', 'Scale_RGT_Benard
                 ', 'Regression_Intercept_RGT_Benard s','3 MIS IPTV_RGT_Benards', '12 MIS IPTV_RGT_Benards' \
        ,'Shape_RGT_Weibull_MLE', 'Scale_RGT_Weibull_ML E','p_Value_RGT_Weibull_ML E','3 MIS IPTV_RGT_Weibull_ML
                 ', '12 MIS IPTV_RGT_Weibull_MLE' \
        ,'Regression Score(%)_Train_RGT_SVM_Lo g','Regression Score(%)_Test_RGT_SVM_Lo g','p_Value_RGT_SVM_Lo
                 g','3 MIS IPTV_RGT_SVM_Lo g','12 MIS IPTV_RGT_SVM_Log' \
        ,'Regression Score(%)_Train_RGT_TFDL_Lo g','Regression Score(%)_Test_RGT_TFDL_Lo g','p_Value_RGT_TFDL_Lo
                 g','3 MIS IPTV_RGT_TFDL_Lo g','12 MIS IPTV_RGT_TFDL_Log'])

    for model in unq_model_wise_complaints['Model'].unique():
        for sub_model in unq_model_wise_complaints['Sub Model'][unq_model_wise_complaints['Model'] == model].unique():
            results = pd.DataFra \
                me(columns=['Mode l','Sub-Model', 'Dummy complaint cod e','Selected_Mont h','Usage_per_Mont h','Shap
                         ', 'Scale', 'p_Value', 'N_DataPoints', 'Regression_Intercep t','Regression Score(%)_Trainin
                         g','Regression Score(%)_Testing' \
                ,'LR_3_MIS_IPT V','LR_12_MIS_IPT V','TF_DL_3_MI S','TF_DL_12_MI S','TF_DL_Training_Scor
                         e','TF_DL_Testing_Scor e','SVM_3_MI S','SVM_12_MI S','SVM_Training_Scor e','SVM_Testing_Score' \
                ,'N_DataPoints_RGT_Cens', 'Regression Score(%)_Train_RGT_KMF', 'Regression Score(%)_Test_RGT_KM
                         ', 'Shape_RGT_KMF', 'Scale_RGT_KMF', 'p_Value_RGT_KMF', 'Regression_Intercept_RGT_KM
                         F','3 MIS IPTV_RGT_KMF', '12 MIS IPTV_RGT_KMF' \
                ,'Regression Score(%)_RGT_Benards', 'Shape_RGT_Benards', 'Scale_RGT_Benard
                         ', 'Regression_Intercept_RGT_Benard s','3 MIS IPTV_RGT_Benards', '12 MIS IPTV_RGT_Benards' \
                ,'Shape_RGT_Weibull_MLE', 'Scale_RGT_Weibull_ML E','p_Value_RGT_Weibull_ML
                         E','3 MIS IPTV_RGT_Weibull_MLE', '12 MIS IPTV_RGT_Weibull_MLE' \
                ,'Regression Score(%)_Train_RGT_SVM_Lo g','Regression Score(%)_Test_RGT_SVM_Lo
                         g','p_Value_RGT_SVM_Lo g','3 MIS IPTV_RGT_SVM_Lo g','12 MIS IPTV_RGT_SVM_Log' \
                ,'Regression Score(%)_Train_RGT_TFDL_Lo g','Regression Score(%)_Test_RGT_TFDL_Lo
                         g','p_Value_RGT_TFDL_Lo g','3 MIS IPTV_RGT_TFDL_Lo g','12 MIS IPTV_RGT_TFDL_Log'])

            raw_data_prepro_temp = raw_data_prep \
                ro[(raw_data_prepro['Model'] == model) & (raw_data_prepro['Sub Model'] == sub_model)]
            # raw_data_prepro_temp = raw_data_prepro_temp[raw_data_prepro_temp['Dummy complaint code'] in unq_model_wise_complaints['Dummy complaint code']]

            results['Dummy complaint code'] = unq_model_wise_complaints['Dummy complaint code
                '][(unq_model_wise_complaints['Model'] == model)
                            & (unq_model_wise_complaints['Sub Model'] == sub_model)].unique()
            results['Model'] = model
            results['Sub-Model'] = sub_model

            # Selecting Best batch basis the Modal Values
            if batch_selection == 'y':
                for complaint in results['Dummy complaint code']:
                    results['Selected_Month'][results['Dummy complaint code'] == complaint] \
                    = raw_data_prepro_temp['Production Month'
                        ][raw_data_prepro_temp['Dummy complaint code'] == complaint].mode()[0]
            ele :
                for complaint in results['Dummy complaint code']:
                    results['Selected_Month'][results['Dummy complaint code'] == complaint] = "All Months"

            for complaint in results['Dummy complaint code']:
                # print(complaint)
                # Checking for No. of DataPoints to skip the analysis for less frequent complaints
                # print(results['Selected_Month'][results['Dummy complaint code'] == complaint].unique()[0])

                if batch_selection == 'y':
                    complaint_data = raw_data_prepro_temp[(raw_data_prepro_temp['Dummy complaint code'] == complaint) & \
                                                          (raw_data_prepro_temp['Production Month']
                                                           == results['Selected_Month'
                                                               ][results['Dummy complaint code'] == complaint].unique(
                                                               )[0])]
                ele :
                    complaint_data = raw_data_prepro_temp[(raw_data_prepro_temp['Dummy complaint code'] == complaint)]
                    # print(complaint_data['Sales Qty'])

                # Filtering outliers through 2 sigma norma distribution
                complaint_data = complaint_data[complaint_data['Sales Qty'] > 30]
                # sales_z_scores = np.abs((complaint_data['Sales Qty'] - complaint_data['Sales Qty'].mean()) / complaint_data['Sales Qty'].std())
                # Define threshold for outliers (e.g., Z-score > 2 , 2.5 , 3)
                # threshold = 2
                # Filter out rows with Z-score > threshold
                # complaint_data = complaint_data[sales_z_scores <= threshold]
                # print("\nDataFrame after removing outliers:")
                # print(complaint_data['Sales Qty'])

                # Checking if sufficient no. of complaints are available for further processing..
                print(complaint_data.shape[0])
                complaint_data = complaint_data.sort_values('Kilometers')
                complaint_data['Cumulative_IPTV'] = complaint_data['IPTV'].cumsum( )
                complaint_data['Cumulative_Hazard'] =  complaint_data['Cumulative_IP T V']/1000
                complaint_data['Log_Kilometers'] = np.log(complaint_data['Kilometers'])
                complaint_data['Log_Cum._Hazard'] = np.log(complaint_data['Cumulative_Hazard'])

                if complaint_data.shape[0] > 6:
                    complaint_usage = round(np.sum(complaint_data['Kilometer s '])/np.sum(complaint_data['Month']))

                    if hzd_ck h =='y':
                        # complaint_data['Log_Month'] = np.log(complaint_data['Month'])
                        # print(complaint_data.shape)
                        # print(results['Selected_Month'][results['Dummy complaint code'] == complaint].unique()[0])
                        # The target(s) (dependent variable) is 'log price'
                        targets = complaint_data['Log_Cum._Hazard']
                        inputs = complaint_data[['Log_Kilometers']]

                        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=
                                                                            0.2, random_state=42)

                        # Create a linear regression object
                        reg = LinearRegression()
                        # Fit the regression with the scaled TRAIN inputs and targets
                        reg.fit(in puts,targets)
                        # y_hat = reg.predict(inputs)
                        # print(reg.coef_[0],    np.exp(-reg.intercept_/reg.coef_[0]),        f_regression(inputs,targets)[1][0],         complaint_data.shape[0])
                        results['Usage_per_Month'][results['Dummy complaint code'] == complaint] = complaint_usage
                        results['Regression Score(%)_Traini
                            ng'][results['Dummy complaint code'] == complaint] = reg.score(X_t rain,y_t r ain)*100
                        results['Regression Score(%)_Testi
                            ng'][results['Dummy complaint code'] == complaint] = reg.score(X_ test,y_ t est)*100
                        results['Shape'][results['Dummy complaint code'] == complaint] = reg.coef_[0]
                        results['Regression_Intercept'][results['Dummy complaint code'] == complaint] = -reg.intercept_
                        results['Scale'][results['Dummy complaint code'] == complaint] = np \
                            .exp(-reg.inter c ept_/reg.coef_[0])
                        results['p_Value'][results['Dummy complaint code'] == complai
                        t] = f_regression(in puts,targets)[1][0]
                        results['N_DataPoints'][results['Dummy complaint code'] == complaint] = complaint_data.shape[0]
                        results['LR_3_MIS_IPTV'][results['Dummy complaint code'] == complaint] = (1-(math.
                            exp(-math.po w ((3*complaint_u s age/np.exp(-reg.interc e pt_/reg.coef_[0
                                      ])),reg.coef_[0] ) )))*1000
                        results['LR_12_MIS_IPTV'][results['Dummy complaint code'] == complaint] = (1-(math.ex
                            p(-math.pow(( 1 2*complaint_usa g e/np.exp(-reg.intercep t _/reg.coef_[0])
                                      ),reg.coef_[0])) ) )*1000

                        SVM_Pred_3_MIS, SVM_Pred_12_MIS, score_train, score_test = SVM_Model.SVM_Hazard_Log(X_train, X_test
                                                                                                  , y_train, y_te
                                                                                                  t, complaint_usage)
                        results['SVM_3_MIS'][results['Dummy complaint code'] == complaint] = SVM_Pred_3_MIS
                        results['SVM_12_MIS'][results['Dummy complaint code'] == complaint] = SVM_Pred_12_MIS
                        results['SVM_Training_Score'][results['Dummy complaint code'] == complaint] = score_tr a in*100
                        results['SVM_Testing_Score'][results['Dummy complaint code'] == complaint] = score_t e st*100
                        # results['TF_DL_Accuracy'][results['Dummy complaint code'] == complaint] = accuracy

                    if rgt_cens == 'y':
                        if batch_selection == 'y':
                            complaint_FC_Data = raw_data_prepro_temp[['Chassis no .','Kilometers'
                                ]][(raw_data_prepro_temp['Dummy complaint code'] == complaint) & \
                                                                                                   (raw_data_prepro_temp['Production Month'] == results['Selected_Month
                                    '][results['Dummy complaint code'] == complaint])].sort_valu \
                                es(['Kilometer s','Chassis no.'])
                        ese :
                            complaint_FC_Data = raw_data_prepro_temp[['Chassis no .','Kilometers'
                                ]][raw_data_prepro_temp['Dummy complaint code'] == complaint].sort_valu \
                                es(['Kilometer s','Chassis no.'])

                        complaint_FC_Data.rena \
                            me(columns={'Chassis no .':'Chassis No.', 'Kilometer s':'Failure_Cumulative_KM'}, inplace=True)

                        # Failure Kms calculation from the cumulative Kilometers
                        complaint_FC_Data['Failure Kms'] = complaint_FC_Data.groupby(['Chassis No.'
                            ])['Failure_Cumulative_KM'].diff().fillna(complaint_FC_Data['Failure_Cumulative_KM'])
                        complaint_FC_Data['F/S'] = 1
                        # Creating the Dataframe which contains only the max. Kms data fro unique chassis from the above data
                        complaint_FC_Data_max_only = complaint_FC_Data.groupby('Chassis No.
                            ')['Failure_Cumulative_KM'].max().reset_index()
                        complaint_FC_Data_max_only.rena \
                            me(coluns = {'Failure_Cumulative_KM': 'Last_Failure_Cumulative_KM'}, inplace=True)
                        # Complaint_FC_Data.sort_values(['Failure_Cumulative_KM','Chassis No.'],ascending=False)
                        # Complaint_FC_Data.reset_index(inplace=True)

                        complaint_survival_kms = BOBJ_CRM_max_kms.copy()
                        # complaint_survival_kms['Max Failure Kms'] = np.where(complaint_survival_kms['Chassis No.'] == complaint_FC_Data['Chassis No.'],complaint_FC_Data['Last Service KM'], 0)

                        # Creating the Survival_falure Data with including the Last Failure data from BOBJ to CRM
                        complaint_survival_kms = complaint_survival_kms.merge(complaint_FC_Data_max_on
                                                                              y, left_on='Chassis No
                                                                              ', right_on='Chassis No.', how='left')
                        complaint_survival_kms['Last_Failure_Cumulative_KM'].fillna(0, inplace=True)

                        # Survival Calculcation from the data
                        complaint_survival_kms['Survival_Kms'] = complaint_survival_kms['Last Service KM'] - complaint_survival_kms['Last_Failure_Cumulative_KM']

                        # Mapping the survival to 0 for further analysis in stacked data
                        complaint_survival_kms['F/S'] = 0

                        # Failure/ Survival Stacked Data..
                        FC_Stacked = pd.DataFrame(columns=['FC_Km s','F/S'])
                        FC_Stacked['FC_Kms'] = pd.conc \
                            at([complaint_survival_kms['Survival_Kms'], complaint_FC_Data['Failure Kms']])
                        FC_Stacked['F/S'] = pd.concat([complaint_survival_kms['F/S'], complaint_FC_Data['F/S']])
                        FC_Stacked.sort_values(['FC_Kms'], inplce = True)
                        FC_Stacked.reset_index(inplce = True)
                        FC_Stacked.drop(['index'], ais = 1, inplace=True)
                        FC_Stacked['Units_at_Risk'] = len(FC_Stacked) - np.arange(0, len(FC_Stacked))

                        # Kaplan Meire Ranking of Data for Prediction
                        kmf = lifelines.KaplanMeierFitter()
                        kmf.fit(FC_Stacked['FC_Kms'], event_observed=FC_Stacked['F/S '],alpha=0.05)

                        reg_kmf = LinearRegression()
                        # Fit the regression with the scaled TRAIN inputs and targets

                        # Changing the initial zero CFD values to very small values as 10^(-6) to avoid "-inf' error after log transformations
                        kmf.cumulative_density_['KM_estimate'][kmf.cumulative_density_['KM_estimate'] == 0] = 0.000001
                        kmf.timeline[kmf.timeline == 0] = 0.000001

                        inputs = np.log(kmf.timeline).reshape(len(kmf.timelin e),1)
                        targets = np.log(np.lo g (1 / (1-kmf.cumulative_density_['KM_estimate'])))
                        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0
                                                                            2, random_state=42)
                        reg_kmf.fit(X_train, y_train)

                        results['Regression Score(%)_Train_RGT_KMF
                            '][results['Dummy complaint code'] == complaint] = reg_kmf.score(X_train, y_tra i n)*100
                        results['Regression Score(%)_Test_RGT_KMF
                            '][results['Dummy complaint code'] == complaint] = reg_kmf.score(X_test, y_te s t)*100
                        results['Shape_RGT_KMF'][results['Dummy complaint code'] == complaint] = reg_kmf.coef_[0]
                        results['Regression_Intercept_RGT_KMF
                            '][results['Dummy complaint code'] == complaint] = -reg_kmf.intercept_
                        results['Scale_RGT_KMF'][results['Dummy complaint code'] == complaint] = np.e \
                            xp(-reg_kmf.interce p t_/reg_kmf.coef_[0])
                        results['p_Value_RGT_KMF'][results['Dummy complaint code'] == complaint] = f_regression(inpu ts,targets)[1][0]
                        results['N_DataPoints_RGT_Cens'][results['Dummy complaint code'] == complaint] = complaint_FC_Data.shape[0]
                        results['3 MIS IPTV_RGT_KMF'][results['Dummy complaint code'] == complaint] = (1-(math.ex
                            p(-math.pow( ( 3*complaint_usa g e/np.exp(-reg_kmf.intercep t _/reg_kmf.coef_[0])
                                      ),reg_kmf.coef_[0])) ) )*1000
                        results['12 MIS IPTV_RGT_KMF'][results['Dummy complaint code'] == complaint] = (1 -(math.exp
                            (-math.pow((1 2 *complaint_usag e /np.exp(-reg_kmf.intercept _ /reg_kmf.coef_[0]))
                                      ,reg_kmf.coef_[0]))) ) *1000

                        # Performing the analysis considering Benards Approx. for Median Ranks calculation
                        Failures_Stacked = FC_Stacked[FC_Stacked['F/S'] == 1]
                        Failures_Stacked['Rank Increment'] = ''
                        Failures_Stacked['Mean Order Number'] = ''
                        Failures_Stacked["Benard's Median Rank"] = ''
                        Failures_Stacked['Log_Kms'] = np.log(Failures_Stacked['FC_Kms'])
                        Failures_Stacked['Log(Log(1/(1-CFD)))'] = ''

                        for i in range(len(Failures_Stacked)):
                            if i == 0:
                                rank_inc = ((len(FC_Stacked ) +1) - 0 ) /(1 + Failures_Stacked['Units_at_Risk'].iloc[i])
                                mean_order_number = 1
                            else :
                                rank_inc = ((len(FC_Stacked ) +1) - Failures_Stacked['Mean Order Number'].iloc
                                    [ i -1] ) /(1 + Failures_Stacked['Units_at_Risk'].iloc[i])
                                mean_order_number = rank_inc + Failures_Stacked['Mean Order Number'].iloc[ i -1]

                            Failures_Stacked['Rank Increment'].iloc[i] = rank_inc
                            Failures_Stacked['Mean Order Number'].iloc[i] = mean_order_number
                            Failures_Stacked["Benard's Median Rank"].iloc[i] = (Failures_Stacked
                                                                                    ['Mean Order Number'].iloc
                                                                                    [i] - 0.3 ) /(len(FC_Stacked) + 0.4)
                            Failures_Stacked['Log(Log(1/(1-CFD)))'].iloc[i] = math.log \
                                (math.log( 1 /( 1 -Failures_Stacked["Benard's Median Rank"].iloc[i])))

                        reg_benard = LinearRegression()
                        # Fit the regression with the scaled TRAIN inputs and targets
                        reg_benard.fit(Failures_Stacked[['Log_Kms']] ,Failures_Stacked[['Log(Log(1/(1-CFD)))']])

                        results['Regression Score(%)_RGT_Benards']
                            [results['Dummy complaint code'] == complaint] = reg_benard.score \
                            (Failures_Stacked[['Log_Kms']] ,Failures_Stacked[['Log(Log(1/(1-CFD)))']] ) *100
                        results['Shape_RGT_Benards'][results['Dummy complaint code'] == complaint] = reg_benard.coef_[0]
                        results['Regression_Intercept_RGT_Benards']
                            [results['Dummy complaint code'] == complaint] = -reg_benard.intercept_
                        results['Scale_RGT_Benards'][results['Dummy complaint code'] == complaint] = np.exp \
                            (-reg_benard.intercept _ /reg_benard.coef_[0])
                        # results['p_Value'][results['Dummy complaint code'] == complaint] = f_regression(inputs,targets)[1][0]
                        # results['N_DataPoints'][results['Dummy complaint code'] == complaint] = complaint_FC_Data.shape[0]
                        results['3 MIS IPTV_RGT_Benards'][results['Dummy complaint code'] == complaint] = ( 1 -
                            (math.exp
                                (-math.pow(( 3 *complaint_usag e /np.exp(-reg_benard.intercept _ /reg_benard.coef_[0]))
                                          ,reg_benard.coef_[0]))) ) *1000
                        results['12 MIS IPTV_RGT_Benards'][results['Dummy complaint code'] == complaint] = ( 1 -
                            (math.exp
                                (-math.pow((1 2 *complaint_usag e /np.exp(-reg_benard.intercept _ /reg_benard.coef_[0]))
                                          ,reg_benard.coef_[0]))) ) *1000

                        # Performing the analysis considering WiebullFitter()
                        wbf = lifelines.WeibullFitter()
                        in_out_df = FC_Stacked[['FC_Kms' ,'F/S']]
                        in_out_df['FC_Kms'][in_out_df['FC_Kms'] == 0] = 0.000001
                        in_out_df['F/S'][in_out_df['F/S'] == 0] = 0.000001
                        wbf.fit(in_out_df['FC_Kms'], event_observed=in_out_df['F/S'] ,alpha=0.05)

                        # results['Regression Score(%)'][results['Dummy complaint code'] == complaint] = reg_benard.score(inputs,targets)*100
                        results['Shape_RGT_Weibull_MLE'][results['Dummy complaint code'] == complaint] = wbf.rho_
                        # results['Regression_Intercept'][results['Dummy complaint code'] == complaint] = -reg_benard.intercept_
                        results['Scale_RGT_Weibull_MLE'][results['Dummy complaint code'] == complaint] = wbf.lambda_
                        results['p_Value_RGT_Weibull_MLE'][results['Dummy complaint code'] == complaint] = f_regression(in_out_df[['FC_Kms']] ,in_out_df[['F/S']])[1][0]
                        results['3 MIS IPTV_RGT_Weibull_MLE'][results['Dummy complaint code'] == complaint] = ( 1 -
                            (math.exp(-math.pow(( 3 *complaint_usag e /wbf.lambda_) ,wbf.rho_))) ) *1000
                        results['12 MIS IPTV_RGT_Weibull_MLE'][results['Dummy complaint code'] == complaint] = ( 1 -
                            (math.exp(-math.pow((1 2 *complaint_usag e /wbf.lambda_) ,wbf.rho_))) ) *1000

                        # SVR Model
                        SVM_Pred_3_MIS, SVM_Pred_12_MIS, score_train, score_test = SVM_Model.SVM_Hazard_Log(X_train, X_test, y_train, y_test, complaint_usage)
                        results['Regression Score(%)_Train_RGT_SVM_Log']
                            [results['Dummy complaint code'] == complaint] = score_train
                        results['Regression Score(%)_Test_RGT_SVM_Log']
                            [results['Dummy complaint code'] == complaint] = score_test
                        results['3 MIS IPTV_RGT_SVM_Log'][results['Dummy complaint code'] == complaint] = SVM_Pred_3_MIS
                        results['12 MIS IPTV_RGT_SVM_Log']
                            [results['Dummy complaint code'] == complaint] = SVM_Pred_12_MIS

                        # TF DL Model
                        tf_dl_Predictions_3_MIS, tf_dl_Predictions_12_MIS, mape_train, mape_test = tf_Hazard_log \
                            (X_train, X_test, y_train, y_test, complaint_usage, 'y')
                        results['Regression Score(%)_Train_RGT_TFDL_Log']
                            [results['Dummy complaint code'] == complaint] = mape_train
                        results['Regression Score(%)_Test_RGT_TFDL_Log']
                            [results['Dummy complaint code'] == complaint] = mape_test
                        results['3 MIS IPTV_RGT_TFDL_Log']
                            [results['Dummy complaint code'] == complaint] = tf_dl_Predictions_3_MIS
                        results['12 MIS IPTV_RGT_TFDL_Log']
                            [results['Dummy complaint code'] == complaint] = tf_dl_Predictions_12_MIS
                try:
                    complaints_selected_df = pd.concat([complaints_selected_df ,complaint_data])
                except:
                    complaints_selected_df = complaint_data
            results_final_temp = pd.concat([results_final_temp ,results])
            # complaints_selected_df = pd.concat([complaints_selected_df,complaint_data])

    if batch_selection = ='y':
        results_final_temp['Selected_Month'] = pd.to_datetime(results_final_temp['Selected_Month']).dt.date
    # results.to_excel('results_temp.xlsx')
    que.put([complaints_selected_df ,results_final_temp])
    print('Item_sent to Que')
    return