import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from collections import defaultdict
import math

class LinearRegressor:

    def __init__(self, continuous=True):
        self.final_continuous = pd.read_csv('finalData.csv')
        self.final_discrete = pd.read_csv('discreteData.csv')
        self.data = self.final_continuous if continuous else self.final_discrete
        self.loadData()

    def loadData(self):
        featuresStandard = ['heightinchestotal', 'weight', 'fortyyd', 'vertical', 'bench', 'twentyss', 'threecone', 'broad',
                    'games', 'rushingAtt', 'rushingYds', 'rushingAvg', 'rushingTD', 'passCmp',
                    'passPct', 'passYds', 'passTD', 'passInt', 'passRate', 'recYds', 'recAtt', 'recTD', 'recAvg', 'soloTackles',
                    'tackleAssists', 'totalTackles', 'sacks', 'ints', 'intTDs', 'passDef', 'fumbles', 'fumblesForced']
        featuresDiscrete = ['confACC','confPac-12','confUnknown','confSEC','confBig 12','confBig Ten','confAmerican',
                    'confBig East','confPac-10','confMAC','confSun Belt','confMWC','confWAC','confCUSA','confInd','confSouthern','confMVC','confPac-8',
                    'confBig West','confSWC','confSouthland','confBig 8','confSWAC','heightinchestotalNone','heightinchestotalQ1','heightinchestotalQ2',
                    'heightinchestotalQ3','heightinchestotalQ4','weightNone','weightQ1','weightQ2','weightQ3','weightQ4','fortyydNone','fortyydQ1','fortyydQ2',
                    'fortyydQ3','fortyydQ4','verticalNone','verticalQ1','verticalQ2','verticalQ3','verticalQ4','benchNone','benchQ1','benchQ2','benchQ3',
                    'benchQ4','twentyssNone','twentyssQ1','twentyssQ2','twentyssQ3','twentyssQ4','threeconeNone','threeconeQ1','threeconeQ2','threeconeQ3',
                    'threeconeQ4','broadNone','broadQ1','broadQ2','broadQ3','broadQ4','gamesNone','gamesQ1','gamesQ2','gamesQ3','gamesQ4','rushingAttNone',
                    'rushingAttQ1','rushingAttQ2','rushingAttQ3','rushingAttQ4','rushingYdsNone','rushingYdsQ1','rushingYdsQ2','rushingYdsQ3','rushingYdsQ4',
                    'rushingAvgNone','rushingAvgQ1','rushingAvgQ2','rushingAvgQ3','rushingAvgQ4','rushingTDNone','rushingTDQ1','rushingTDQ2','rushingTDQ3','rushingTDQ4'
                    ,'passCmpNone','passCmpQ1','passCmpQ2','passCmpQ3','passCmpQ4','passPctNone','passPctQ1','passPctQ2','passPctQ3','passPctQ4','passYdsNone','passYdsQ1',
                    'passYdsQ2','passYdsQ3','passYdsQ4','passTDNone','passTDQ1','passTDQ2','passTDQ3','passTDQ4','passIntNone','passIntQ1','passIntQ2','passIntQ3','passIntQ4','passRateNone',
                    'passRateQ1','passRateQ2','passRateQ3','passRateQ4','recYdsNone','recYdsQ1','recYdsQ2','recYdsQ3','recYdsQ4','recAttNone','recAttQ1','recAttQ2','recAttQ3',
                    'recAttQ4','recTDNone','recTDQ1','recTDQ2','recTDQ3','recTDQ4','recAvgNone','recAvgQ1','recAvgQ2','recAvgQ3','recAvgQ4','soloTacklesNone','soloTacklesQ1',
                    'soloTacklesQ2','soloTacklesQ3','soloTacklesQ4','tackleAssistsNone','tackleAssistsQ1','tackleAssistsQ2','tackleAssistsQ3','tackleAssistsQ4','totalTacklesNone',
                    'totalTacklesQ1','totalTacklesQ2','totalTacklesQ3','totalTacklesQ4','sacksNone','sacksQ1','sacksQ2','sacksQ3','sacksQ4','intsNone','intsQ1','intsQ2','intsQ3',
                    'intsQ4','intTDsNone','intTDsQ1','intTDsQ2','intTDsQ3','intTDsQ4','passDefNone','passDefQ1','passDefQ2','passDefQ3','passDefQ4','fumblesNone','fumblesQ1',
                    'fumblesQ2','fumblesQ3','fumblesQ4','fumblesForcedNone','fumblesForcedQ1','fumblesForcedQ2','fumblesForcedQ3','fumblesForcedQ4']
        self.features = featuresStandard if self.data.equals(self.final_continuous)  else featuresDiscrete
        self.shouldNormalize = True if self.features == featuresStandard else False
        self.draftValues, self.draftPositions = self.get_player_draft_info()
        self.positions = self.get_positions()
        self.combineFeatures = self.data[['name', 'pos', 'year'] + self.features]
        self.positionMaps = self.get_position_maps()

    # get player values/draft positions based on 'nfl_draft' data
    def get_player_draft_info(self):
        valuesMap = {}
        pickMap = {}
        MAX_VALUE = 256 # number of draft picks + 1 (7 rounds * 32 picks + 32 compensatory + 1)
        for index, row in self.data.iterrows():
            primaryKey = (row['name'], row['year'])
            valuesMap[primaryKey] = (MAX_VALUE - row['pick'])
            pickMap[primaryKey] = row['pick']
        return valuesMap, pickMap

    # Creates list of positions to partition by
    def get_positions(self):
        positionsSet = set()
        for position in self.data['pos']:
            positionsSet.add(position)
        return list(positionsSet)


    def get_position_maps(self):
        positionMaps = {}
        for position in self.positions:
            positionMaps[position] = self.combineFeatures[self.combineFeatures['pos'] == position]
        return positionMaps

    # Map players to their true and predicted draft positions
    def get_relative_error(self, results, test_year):
        trueDraftMap = {}  # true pick number
        predictedDraftMap = {}  # relative pick based on regression
        i = 0
        for index, row in results.iterrows():
            trueDraftMap[row['name']] = self.draftPositions[(row['name'], test_year)]  # need primary key to get draft positions
            predictedDraftMap[row['name']] = i
            i += 1

        # Compute absolute_error based on relative draft positions
        i = 0
        absolute_error = 0
        errors = {}
        for key, value in sorted(trueDraftMap.iteritems(), key=lambda (k, v): (v, k)):
            errors[key] = math.fabs(i - predictedDraftMap[key])
            absolute_error += errors[key]
            i += 1

        # Add the errors to the results Dataframe (for visualization)
        results['error'] = 0.0
        for index, row in results.iterrows():
            results.at[index, 'error'] = errors[row['name']]

        # average the error by the total number of players
        return absolute_error / float(len(results))

    # build X matrix and Y values
    def build_data_arrays(self, TEST_YEAR):
        X = defaultdict(list)
        Y = defaultdict(list)
        xTrain = defaultdict(list)
        yTrain = defaultdict(list)
        xTest = defaultdict(list)
        yTest = defaultdict(list)
        names = defaultdict(list)
        for position in self.positionMaps:
            for index, row in self.positionMaps[position].iterrows():
                X[position].append(row[3:])
                Y[position].append(self.draftValues[(row['name'], row['year'])])
                if row['year'] == TEST_YEAR:
                    xTest[position].append(row[3:])
                    yTest[position].append(self.draftValues[(row['name'], row['year'])])
                    names[position].append(row['name'])
                if not row['year'] == TEST_YEAR:
                    xTrain[position].append(row[3:])
                    yTrain[position].append(self.draftValues[(row['name'], row['year'])])
        return X, Y, xTrain, yTrain, xTest, yTest, names

    def fit_and_predict(self, TEST_YEAR, regularization=True):
        X, Y, xTrain, yTrain, xTest, yTest, names = self.build_data_arrays(TEST_YEAR)
        seed = 7
        np.random.seed(seed)
        if regularization:
            predictor = linear_model.RidgeCV(alphas=[0.1, 1.0, 10], fit_intercept=True, normalize=self.shouldNormalize)
        else:
            predictor = linear_model.LinearRegression(fit_intercept=True, normalize=self.shouldNormalize, copy_X=True, n_jobs=1)
        scores = {}
        relativeError = {}
        coefficients = {}
        output = {}
        for p in self.positions:
            if len(xTrain[p]) > 1 and len(xTest[p]) > 1:
                predictor.fit(np.array(xTrain[p]), np.array(yTrain[p]))
                coefficients[p] = pd.DataFrame(zip(self.features, predictor.coef_), columns = ['feature', 'coefficient']).sort_values(by=['coefficient'], ascending=False)
                prediction = predictor.predict(np.array(xTest[p]))
                output[p] = pd.DataFrame(zip(names[p], prediction), columns = ['name', 'value']).sort_values(by=['value'], ascending=False)
                scores[p] = (mean_squared_error(np.array(yTest[p]), np.array(prediction)),
                             r2_score(np.array(yTest[p]), np.array(prediction)))
                relativeError[p] = self.get_relative_error(output[p], TEST_YEAR)
        return output


