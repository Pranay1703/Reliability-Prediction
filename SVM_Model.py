from sklearn.svm import SVR
import pandas as pd

def SVM_Hazard_Log(X_train, X_test, y_train, y_test, usage):
    model = SVR(kernel='rbf', C=100, gamma=0.001)
    model.fit(X_train,y_train)
    #print("SVM Score :",model.score(X,y))
    #model.predict(X)
    Pred_3_MIS = np.exp(model.predict(pd.DataFrame(np.log([3*usage])))[0])*1000
    Pred_12_MIS = np.exp(model.predict(pd.DataFrame(np.log([12*usage])))[0])*1000
    return Pred_3_MIS, Pred_12_MIS, model.score(X_train,y_train), model.score(X_test,y_test)