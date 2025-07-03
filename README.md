The Project is a part of a bigger Reliability Prediction program, wherein the statistical part is considered
The data is the Field failure data for various parts/subassemblies, and overall vehicle.
The Program predicts the reliability in terms of IPTV(Issues Per Thousand Vehicles) and expenses incurred, basis the various input parameters as below 
KMs to failure, Months to failure, Zone/Region of Failure, Complaint code, aggregate, production month, production plant location, production batch, variant of vehicle, idle time, repair time etc.
Basis the above parameters, the model identifies the major significant features and then predicts the IPTV.
Multiple ML models are considered for prediction and output is the model, submodel, and complaint wise IPTV wrto time/Kms
Also the results are utilised in the seperate dashboard in powerBI/ Dash app in web.

Predicition_Main is the main file which requires excel file as input data/ further SAP connectivity is there in the Main_app.
