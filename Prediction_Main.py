import Hazard_Main
from multiprocessing import Pool, Process, Queue, freeze_support
import pandas as pd
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')
# Set the Flask logging level to WARNING (or higher)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def iteration_cal(cnt):
    if cnt < 150:
        n_iter = 1
    elif cnt < 300:
        n_iter = 2
    elif cnt < 800:
        n_iter = 4
    elif cnt <= 4:
        n_iter = 4
    elif cnt > 4:
        n_iter = 8
    return(n_iter)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    IPTV_path = input("Enter Cumulative IPTV Path : ")
    raw_data = pd.read_excel(r"{}".format(IPTV_path) ,sheet_name='Formatted'
                             ,dtype={'Sales Month' :'str', 'Production Month' :'str'})

    # Hazard Analysis
    Batch_Selection = input("Want to Select Batch (Type 'y') : ")
    Hazard_check = input("Want to Do Hazard Analysis (Type 'y') : ")
    # Rgt_Cens_check = input("Want to Do Right Censoring Prediction (Type 'y') : ")
    CRM_Path = 'NA'
    if Rgt_Cens_check == 'y':
        CRM_Path = input("Provide the Path for CRM File [Enter Path without Quotes] : ")

    complaints_selected_df, results = Hazard_Main.hazard_main(raw_data, CRM_Path, Batch_Selection.lower(), Hazard_check.lower(), Rgt_Cens_check.lower())

    with pd.ExcelWriter(r"{}".format(IPTV_path), engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
        complaints_selected_df.to_excel(writer, sheet_name='Python_Preprocessing', index=False)
        results.to_excel(writer, sheet_name='Results_Complaints', index=False)