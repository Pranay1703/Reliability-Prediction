import pandas as pd
import multiprocessing
from multiprocessing import Pool, Process, Queue, freeze_support
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')
# Set the Flask logging level to WARNING (or higher)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import multiprocessing
import iterator_Main
import Preprocessing


def hazard_main(raw_data, CRM_Path, batch_selection='y', hzd_ckh='y', rgt_cens='y'):
    raw_data_prepro = Preprocessing.Preprocessing_Hazard(raw_data)
    print(raw_data_prepro.shape)
    results_final = pd.DataFrame(
        columns=['Model', 'Sub-Model', 'Dummy complaint code', 'Selected_Month', 'Usage_per_Month', 'Shape', 'Scale',
                 'p_Value', 'N_DataPoints', 'Regression_Intercept', 'Regression Score(%)_Training',
                 'Regression Score(%)_Testing' \
            , 'LR_3_MIS_IPTV', 'LR_12_MIS_IPTV', 'TF_DL_3_MIS', 'TF_DL_12_MIS', 'TF_DL_Training_Score',
                 'TF_DL_Testing_Score', 'SVM_3_MIS', 'SVM_12_MIS', 'SVM_Training_Score', 'SVM_Testing_Score' \
            , 'N_DataPoints_RGT_Cens', 'Regression Score(%)_Train_RGT_KMF', 'Regression Score(%)_Test_RGT_KMF',
                 'Shape_RGT_KMF', 'Scale_RGT_KMF', 'p_Value_RGT_KMF', 'Regression_Intercept_RGT_KMF',
                 '3 MIS IPTV_RGT_KMF', '12 MIS IPTV_RGT_KMF' \
            , 'Regression Score(%)_RGT_Benards', 'Shape_RGT_Benards', 'Scale_RGT_Benards',
                 'Regression_Intercept_RGT_Benards', '3 MIS IPTV_RGT_Benards', '12 MIS IPTV_RGT_Benards' \
            , 'Shape_RGT_Weibull_MLE', 'Scale_RGT_Weibull_MLE', 'p_Value_RGT_Weibull_MLE', '3 MIS IPTV_RGT_Weibull_MLE',
                 '12 MIS IPTV_RGT_Weibull_MLE' \
            , 'Regression Score(%)_Train_RGT_SVM_Log', 'Regression Score(%)_Test_RGT_SVM_Log', 'p_Value_RGT_SVM_Log',
                 '3 MIS IPTV_RGT_SVM_Log', '12 MIS IPTV_RGT_SVM_Log' \
            , 'Regression Score(%)_Train_RGT_TFDL_Log', 'Regression Score(%)_Test_RGT_TFDL_Log', 'p_Value_RGT_TFDL_Log',
                 '3 MIS IPTV_RGT_TFDL_Log', '12 MIS IPTV_RGT_TFDL_Log'])

    # count of No. of Processors for Multiprocessing
    cpu_cnt = multiprocessing.cpu_count()

    # Count of Number of Rows in the Final Results with Duplicate removal of Model, Sub Model, Complaint Description....
    unq_model_wise_complaints = raw_data_prepro[['Model', 'Sub Model', 'Dummy complaint code']].drop_duplicates(
        subset=['Model', 'Sub Model', 'Dummy complaint code'])
    cnt = unq_model_wise_complaints.shape[0]
    complaints_selected_df = pd.DataFrame(columns=raw_data_prepro.columns)

    if rgt_cens == 'y':
        raw_CRM_data = pd.read_excel("{}".format(CRM_Path), sheet_name='Filtered')

        # CRM Mas of Last Service Kms
        CRM_max_kms = raw_CRM_data.groupby('Chassis No.')['Last Service KM'].max().reset_index()

        # BOBJ max of Last Service Kms
        BOBJ_max_kms = raw_data_prepro.groupby('Chassis no.')['Kilometers'].max().reset_index()
        BOBJ_max_kms.rename(columns={'Chassis no.': 'Chassis No.', 'Kilometers': 'Last Service KM'}, inplace=True)

        # Calculating Max of Max of Kms from BOBJ and CRM
        BOBJ_CRM_max_kms = pd.concat([CRM_max_kms, BOBJ_max_kms])
        BOBJ_CRM_max_kms = BOBJ_CRM_max_kms.groupby('Chassis No.')['Last Service KM'].max().reset_index()

    else:
        BOBJ_CRM_max_kms = pd.DataFrame()

    # results_final['Model'] = unq_model_wise_complaints['Model']
    # results_final['Sub-Model'] = unq_model_wise_complaints['Sub Model']
    # results_final['Dummy complaint code'] = unq_model_wise_complaints['Dummy complaint code']

    processes = []
    qu_lst = []
    # p_data = bs4.BeautifulSoup(html_data,'lxml')
    n_iter = iteration_cal(cnt)
    m = round(cnt / n_iter)

    print(n_iter, cnt)

    for pr in range(n_iter):
        strt = m * pr
        endd = m * (pr + 1)
        if pr == n_iter:
            endd = cnt
        qu = multiprocessing.Queue()
        # pros_data = p_data.select('tbody[class = "ui-widget-content ui-iggrid-tablebody ui-ig-record ui-iggrid-record"] tr')[strt:endd]
        pros = multiprocessing.Process(target = Iterator_Main.iterator,
                                       args=(unq_model_wise_complaints.iloc[strt:endd], raw_data_prepro, qu,
                                             batch_selection, hzd_ckh, rgt_cens, BOBJ_CRM_max_kms))
        print('Process {} started (Analysing rows {}-{})'.format(pr + 1, strt, endd))
        qu_lst.append(qu)
        processes.append(pros)

    for pr_s in range(n_iter):
        print('Process {} started'.format(pr_s))
        processes[pr_s].start()

    # complaints_selected_df = pd.DataFrame()
    # results = pd.DataFrame()
    for itms in range(n_iter):
        print("Getting Data from Process {}".format(itms + 1))
        item = qu_lst[itms].get()
        print("Data Retrived from Process {}".format(itms + 1))
        if itms == 0:
            complaints_selected_df = item[0]
        else:
            complaints_selected_df = pd.concat([complaints_selected_df, item[0]])
        results_final = pd.concat([results_final, item[1]])

    for pr_e in range(n_iter):
        try:
            processes[pr_e].join()
        except:
            pass

    print('All processes completed')
    return complaints_selected_df, results_final