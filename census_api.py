import json
import requests
import secrets
import pandas as pd
import sqlite3
pd.set_option('display.max_columns', None)

client_key = secrets.CENSUS_API_KEY
CACHE_FILE_NAME = "census_cache.json"
CACHE_DICT = {}

def aggregate_data(name, data_frame, column_index):
    col = int(column_index)
    df = data_frame.iloc[:, 1:col]
    data_frame[name] = df.sum(axis = 1)
    data_frame = data_frame.drop(columns = df)
    return data_frame

def load_cache():
    ''' Opens the cache file if it exists and loads the JSON into
    the CACHE_DICT dictionary.
    if the cache file doesn't exist, creates a new cache dictionary

    Parameters
    ----------
    None

    Returns
    -------
    The opened cache: dict
    '''
    try:
        cache_file = open(CACHE_FILE_NAME, 'r')
        cache_file_contents = cache_file.read()
        cache = json.loads(cache_file_contents)
        cache_file.close()
    except:
        cache = {}
    return cache

def save_cache(cache):
    ''' Saves the current state of the cache to disk

    Parameters
    ----------
    cache_dict: dict
        The dictionary to save

    Returns
    -------
    None
    '''
    cache_file = open(CACHE_FILE_NAME, 'w')
    contents_to_write = json.dumps(cache)
    cache_file.write(contents_to_write)
    cache_file.close()

def construct_unique_key(baseurl, params):
    ''' constructs a key that is guaranteed to uniquely and
    repeatably identify an API request by its baseurl and params


    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    params: dict
        A dictionary of param:value pairs

    Returns
    -------
    string
        the unique key as a string
    '''
    alphabetized_keys = sorted(params.keys())
    param_strings = []
    for key in alphabetized_keys:
        param_strings.append("{}_{}".format(key, params[key]))
    return baseurl + "_" + "_".join(param_strings)

def make_request(baseurl, params):
    '''Make a request to the Web API using the baseurl and params

    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    params: dictionary
        A dictionary of param:value pairs

    Returns
    -------
    dict
        the data returned from making the request in the form of
        a dictionary
    '''
    new_request = requests.get(baseurl, params).json()
    return new_request

def make_request_with_cache(baseurl, get, county, state ):
    '''Check the cache for a saved result for this baseurl+params:values
    combo. If the result is found, return it. Otherwise send a new
    request, save it, then return it.


    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    hashtag: string
        The hashtag to search for
    count: integer
        The number of results you request from Twitter

    Returns
    -------
    dict
        the results of the query as a dictionary loaded from cache
        JSON
    '''
    try:
        with open(CACHE_FILE_NAME) as my_data:
            CACHE_DICT = json.load(my_data)
        my_data.close()
    except:
        CACHE_DICT = {}


    params_dict = {}
    params_dict['get'] = get
    params_dict['for'] = county
    params_dict['in'] = state

    uniq_url = construct_unique_key(baseurl, params_dict)
    # print(uniq_url)

    if uniq_url in CACHE_DICT.keys():
        print("fetching cached data")
    else:
        print("making new request")
        CACHE_DICT[uniq_url] = make_request(baseurl, params_dict)
        with open(CACHE_FILE_NAME, 'w') as outfile:
            outfile.write(json.dumps(CACHE_DICT, indent=2))
        outfile.close()
    return CACHE_DICT[uniq_url]

def PCA_kaiser(in_data):
    """This function performs a PCA for an input dataset with multiple
    independent variables. The Kaiser rule is applied to the resulting
    components. The output is a pandas dataframe with all remaining
    components with and eigenvalue over 1.00."""

    component_cnt = len(in_data.columns)
    X_scaled = StandardScaler().fit_transform(in_data)
    pca = PCA(component_cnt)
    f = pca.fit(X_scaled)
    t = pca.transform(X_scaled)
    PCA_Components = pd.DataFrame(t)
    keep_components = 0
    for eigval in pca.explained_variance_:
        if eigval > 1:
            keep_components = keep_components + 1
    return pd.DataFrame(PCA_Components.iloc[:, 0:keep_components])

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """This function performs a varimax rotation for a set of PCA components.
    the input is a pandas DataFrame containing the component scores, and the
    output is a pandas DataFrame with the rotated scores"""
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(
            dot(Phi.T,asarray(
                Lambda)**3 - (gamma/p) * dot(
                    Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return pd.DataFrame(dot(Phi, R))

def factorLoadings(PCA_table, in_table):
    """This function computes the factor loadings for each component
    in a PCA dataset. The function takes 2 arguments:
        PCA_table - dataframe containing PCA scores
        in_table - original input dataframe used to create the PCA scores

    The output is a dataframe containing the factor loadings for each
    component.
    """
    compare = pd.concat([PCA_table, in_table], axis=1, sort=False)
    corr = compare.corr()
    pca_cols = len(PCA_table.columns)
    return corr.iloc[pca_cols:, :pca_cols]

# def create_db():
#     conn = sqlite3.connect('census_data.sqlite')
#     cur = conn.cursor()
#
#     drop_data_sql = 'DROP TABLE IF EXISTS "Vulnerability_data"'
#     create_vulnerability_data_sql = '''
#         CREATE TABLE IF NOT EXISTS Vulnerability_data(
#             "Id" INTEGER PRIMARY KEY AUTOINCREMENT,
#             "Name" TEXT NOT NULL,
#             "State" TEXT NOT NULL,
#             "County" TEXT NOT NULL,
#             "TotalFemalePopulation" INTEGER NOT NULL,
#             "UnderAge10" INTEGER NOT NULL,
#             "OverAge64" INTEGER NOT NULL,
#             "WithADisability" INTEGER NOT NULL,
#             "PovertyStatus" INTEGER NOT NULL,
#             "Unemployment" INTEGER NOT NULL,
#             "Part-timeWorkers" INTEGER NOT NULL,
#             "Renters" INTEGER NOT NULL,
#             "ReceivePublicAssistance" INTEGER NOT NULL,
#             "SingleParentHouseholds" INTEGER NOT NULL,
#             "NoVehicleAvailable" INTEGER NOT NULL,
#             )
#         '''
#     cur.execute(drop_data_sql)
#     cur.execute(create_vulnerability_data_sql)
#     conn.commit()
#     conn.close()
#
#     create_db()

if __name__ == "__main__":

    CACHE_DICT = load_cache()

    HOST = 'https://api.census.gov/data'
    year = '2016'
    dataset = 'acs/acs5'
    baseurl = "/".join([HOST, year, dataset])

    get_vars = [
    'NAME',
    #Female Population
    'B01001_026E',
    #Under Age 10
    "B01001_003E", "B01001_004E", "B01001_027E","B01001_028E",
    ## Over age 64:
    "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E",
    "B01001_044E", "B01001_045E", "B01001_046E",
    "B01001_047E", "B01001_048E", "B01001_049E",

    # # With a disability:
    "B18101_004E", "B18101_007E", "B18101_010E",
    "B18101_013E", "B18101_016E", "B18101_019E",
    "B18101_023E", "B18101_026E", "B18101_029E",
    "B18101_032E", "B18101_035E", "B18101_038E",

    # # Poverty status:
    "B17020_002E",
    # Unemploymed:
    "C18120_006E",
                    #
    #Part-time workers:
    "B23027_005E", "B23027_010E", "B23027_015E",
    "B23027_020E", "B23027_030E", "B23027_035E",
    #
    # Renters:
    "B25009_010E",
    #
    # Recieve public assistance:
    "B19057_002E",
    #
    # Single parent households:
    "B09005_004E", "B09005_005E",
    #
    # No vehicle available:
    "B08201_002E"
    ]

    predicates = {}
    predicates['get'] = ",".join(get_vars)
    get = ",".join(get_vars) #params: list of varaibles
    county = "county:*" #for: geography of interest
    state = "state:72"

    predicates['get'] = get
    predicates['for'] = county
    predicates['in'] = state
    census = make_request(baseurl=baseurl, params= predicates)

    census_data = make_request_with_cache(baseurl, get, county, state)
    header = census_data[0]


    df = pd.DataFrame(columns = header, data = census_data[1:] )
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)

    df = aggregate_data('TotalFemalePopulation', df, 2)
    df = aggregate_data('UnderAge10', df, 5)
    df = aggregate_data('OverAge64', df, 13)
    df = aggregate_data('WithADisability', df, 13)
    df = aggregate_data('PovertyStatus', df, 2)
    df = aggregate_data('Unemployment', df, 2)
    df = aggregate_data('Part-timeWorkers', df, 7)
    df = aggregate_data('Renters', df, 2)
    df = aggregate_data('ReceivePublicAssistance', df, 2)
    df = aggregate_data('SingleParentHouseholds', df, 3)
    df = aggregate_data('NoVehicleAvailable', df, 2)

    db = sqlite3.connect('census_data.db')
    df.to_sql('census_data', con = db, if_exists = 'replace')
