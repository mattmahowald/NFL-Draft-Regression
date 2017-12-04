import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import defaultdict
import math
import csv
import copy

def featureExtractor(filename, features):
    salaries = pd.read_csv(filename)
    teams = salaries.team.unique()
    years = salaries.year.unique()
    allFeatures = []
    for year in years:
        for team in teams:
            result = []
            sliced = salaries[((salaries['team'] == team) & (salaries['year'] == year))]
            teamFeatures = defaultdict(lambda: [])
            for index, row in sliced.iterrows():                
                pos = row['Pos.']
                cap = row['Cap %']
                teamFeatures[pos].append(cap)
            for (position, count) in features:
                posList = sorted(teamFeatures[position], reverse=True)
                for count in range(count):
                    if len(posList) == 0:
                        result.append(0)
                    else:
                        result.append(posList[0])
                        posList = posList[1:]
            allFeatures.append((team, year, result))
    return allFeatures

def distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1)-np.array(vec2))

def kNNTrain(teamFeatures, draft_order, k):
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    features = []
    output = []
    for (team, year, vec) in teamFeatures:
        if((year, team) in draft_order):
            features.append(vec)
            output.append(draft_order[(year, team)])
    features = np.array(features)
    output = np.array(output)
    output.reshape(1,-1)
    neigh.fit(features, output)
    return neigh 

def kNNPredict(neigh, featureVec):
    return neigh.predict_proba(featureVec)

def get_draft_position(filename):
    '''Returns a dict mapping team to position selected in the first round'''
    teams = {'PIT':'pittsburgh-steelers', 'CIN':'cincinnati-bengals', 'BAL':'baltimore-ravens', 'CLE':'cleveland-browns',
         'NWE':'new-england-patriots', 'BUF':'buffalo-bills', 'MIA':'miami-dolphins', 'NYJ':'new-york-jets',
         'TEN':'tennessee-titans', 'HOU':'houston-texans', 'IND':'indianapolis-colts', 'JAX':'jacksonville-jaguars',
         'KAN':'kansas-city-chiefs', 'OAK':'oakland-raiders', 'DEN':'denver-broncos', 'SDG':'san-diego-chargers',
         'GNB':'green-bay-packers', 'MIN':'minnesota-vikings', 'DET':'detroit-lions', 'CHI':'chicago-bears',
         'DAL':'dallas-cowboys', 'NYG':'new-york-giants', 'PHI':'philadelphia-eagles', 'WAS':'washington-redskins',
         'CAR':'carolina-panthers', 'ATL':'atlanta-falcons', 'NOR':'new-orleans-saints', 'TAM':'tampa-bay-buccaneers',
         'SEA':'seattle-seahawks', 'ARI':'arizona-cardinals', 'STL':'st.-louis-rams', 'SFO':'san-francisco-49ers'}
    draft = pd.read_csv(filename)
    draft_order = {}
    for _, row in draft.iterrows():
        if row['Tm'] in teams:
            draft_order[(row['Year'], teams[row['Tm']])] = row['Position Standard']
        else:
            print 'not found', row
    return draft_order

def predict_draft_position(feature_vector, draft_picks, teamFeatures):
    kTeams = kNN(feature_vector, teamFeatures, 10)
    similar_drafts = defaultdict(int)
    for team_data in kTeams:
        team, year, distance = team_data
        if (year, team) in draft_picks:
            similar_drafts[draft_picks[(year, team)]] += 1
    return similar_drafts

DEFAULT_WEIGHTS = [1, 1, 4, 3, 1, 2, 2, 2, 2, 3, 2]
def get_draft_probability_matrix(k=5, weights=DEFAULT_WEIGHTS):
    positions =['QB', 'RB', 'DB', 'LB', 'C', 'DE', 'DT', 'G', 'TE', 'WR', 'T'] 
    features = [(positions[idx], weights[idx]) for idx in range(len(positions))]
    trainFeatures = featureExtractor('data/salaries.train.csv', features)
    devFeatures  = featureExtractor('data/salaries.dev.csv', features)
    testFeatures = featureExtractor('data/salaries.test.csv', features)

    draftTrain = get_draft_position('data/nfldraft.train.csv')
    draftDev = get_draft_position('data/nfldraft.dev.csv')
    draftTest = get_draft_position('data/nfldraft.test.csv')

    data = {}
    for i in range(32):
        neigh = kNNTrain(trainFeatures, draftTrain, k)
        probabilities = kNNPredict(neigh, testFeatures[i][2])
        probabilityDict = {features[idx][0]: probability for idx, probability in enumerate(probabilities[0])}
        data[testFeatures[i][0]] = probabilityDict
    return data
        
DEFAULT_RD_BARRIERS = [1, 2, 3, 6, 11, 16, 20, 33]
def positionMLE(filename, barriers=DEFAULT_RD_BARRIERS):
    positions =['QB', 'RB', 'DB', 'LB', 'C', 'DE', 'DT', 'G', 'TE', 'WR', 'T'] 
    laplace_map = defaultdict(int)
    for position in positions:
        laplace_map[position] = 1
    pick_likelihoods = {}
    for idx in range(len(barriers)-1):
        map_copy = copy.deepcopy(laplace_map)
        for rk in range(barriers[idx], barriers[idx+1]):
            pick_likelihoods[rk] = map_copy
            
    draft_order_by_position = pd.read_csv(filename)
    for idx, row in draft_order_by_position.iterrows():
        rk = row[1]
        pos = row[2]
        pick_likelihoods[rk][pos]+=1
        
    ### normalize
    for pick in barriers[:-1]:
        d = pick_likelihoods[pick]
        factor=1.0/sum(d.itervalues())
        for k in d:
            d[k] = d[k]*factor
    
    return pick_likelihoods
    