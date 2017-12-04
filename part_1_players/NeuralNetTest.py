from Regressor import Regressor

regressor = Regressor()
outputLR = regressor.fit_and_predict(2015)
outputNN = regressor.fit_and_predict(2015)

print outputLR
print outputNN